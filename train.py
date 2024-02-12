
import os
import torch
from code.parser_args import get_args
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from code.model import Model
from code.dataset import CustomDataset, Collate_fn
from code.evaluate import evaluate, test
from transformers import BertTokenizer, AutoTokenizer, set_seed
from collections import defaultdict
from prettytable import PrettyTable

def get_kl_loss(one, target):
    # bs x dim
    m = nn.LogSoftmax(dim=1)
    kl = nn.KLDivLoss(reduction='batchmean', log_target=True)(input=m(one), target=m(target))
    return kl

def get_ct_loss(one, two, device):
    # bs x dim
    cos_sim = nn.CosineSimilarity(dim=-1)(one.unsqueeze(1), two.unsqueeze(0))  # bs x bs
    loss_fun = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(device)
    loss = loss_fun(cos_sim/0.05, labels)
    return loss

def calc_loss(out_s, labels, device, step, temp=0.05, ancher=None, model=None):
    """
    out_s: (bs*2) x dim
    labels: bs
    """
    out = out_s.view(3, -1, out_s.size(-1))
    bs = out.size(1)

    z1 = out[0]  # bs x dim
    z2 = out[1]
    z3 = out[2]

    cos_sim = nn.CosineSimilarity(dim=-1)(z1.unsqueeze(1), z2.unsqueeze(0))
    z1_z3_cos = nn.CosineSimilarity(dim=-1)(z1.unsqueeze(1), z3.unsqueeze(0))
    cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)/0.05  # bs x (bs*2)

    labels = torch.arange(cos_sim.size(0)).long().to(device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)


    return loss, [loss.item()]

def train():
    args = get_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    list_sample = list()
    with open('data/train/train_tri.json', encoding='utf-8') as f:
        list_line = f.readlines()
        import json
        for json_str in list_line:
            list_sample.append(json.loads(json_str))


    tokenizer = AutoTokenizer.from_pretrained('model_para/xlmr-base')
    collate_fn = Collate_fn(tokenizer)
    train_dataset = CustomDataset(list_sample=list_sample)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=collate_fn, num_workers=1)

    total_steps = len(train_dataloader) * args.epoch

    num_train_optimization_steps = int(
        len(train_dataset) / args.batch_size) * args.epoch

    model = Model(my_device=device)

    if torch.cuda.is_available():        
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1,
                                                num_training_steps=total_steps)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)

    best_res = defaultdict(float)

    def eval_now(step):
        print('-------------------------------         EVAL         ---------------------------------')
        result = evaluate(model, args)
        # print(result)
        tatoeba = result['Tatoeba']['validation']['cmn-eng']['f1'] * 100
        bucc = result['BUCC']['validation']['zh-en']['f1'] * 100

        stsb = result['STSBenchmark']['validation']['cos_sim']['spearman'] * 100

        atec = result['ATEC']['validation']['cos_sim']['spearman'] * 100
        lcqmc = result['LCQMC']['validation']['cos_sim']['spearman'] * 100
        bq = result['BQ']['validation']['cos_sim']['spearman'] * 100
        pawsx = result['PAWSX']['validation']['cos_sim']['spearman'] * 100
        stsb_zh = result['STSB']['validation']['cos_sim']['spearman'] * 100


        mean_zh = (atec+lcqmc+bq+pawsx+stsb_zh) / 5
        mean_en = stsb
        mean = (mean_en + mean_zh)/2
        if not best_res or mean > best_res['mean']:
            best_res['mean'] = mean
            best_res['tatoeba'] = tatoeba
            best_res['bucc'] = bucc

            best_res['stsb'] = stsb

            best_res['atec'] = atec
            best_res['lcqmc'] = lcqmc
            best_res['bq'] = bq
            best_res['pawsx'] = pawsx
            best_res['stsb_zh'] = stsb_zh
            best_res['mean_zh'] = mean_zh

            best_res['step'] = step

            os.makedirs('result/', exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join('result/', "T2S.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        tb = PrettyTable(["Tatoeba", "BUCC", "Avg."])
        tb.add_row([tatoeba, bucc, (tatoeba+bucc)/2])
        tb.float_format = "4.2"
        print(tb)
        print()

        tb = PrettyTable(["", "STSB", "ATEC", "LCQMC", "BQ", "PAWSX", "STS-Z", "Avg.zh", "Avg.", "step."])
        tb.add_row(["Now", stsb, atec, lcqmc, bq, pawsx, stsb_zh, mean_zh, mean, step])
        tb.add_row(["Best", best_res['stsb'], best_res['atec'], best_res['lcqmc'], best_res['bq'], best_res['pawsx'],
                    best_res['stsb_zh'], best_res['mean_zh'], best_res['mean'], best_res['step']])
        tb.float_format = "4.2"
        print(tb)
        print()

    def test_now():
        print('-------------------------------         TEST         ---------------------------------')
        result = test(model, args)

        def print_result():
            tatoeba = result['Tatoeba']['test']['cmn-eng']['f1'] * 100
            bucc = result['BUCC']['test']['zh-en']['f1'] * 100

            stsb = result['STSBenchmark']['test']['cos_sim']['spearman'] * 100
            sts12 = result['STS12']['test']['cos_sim']['spearman'] * 100
            sts13 = result['STS13']['test']['cos_sim']['spearman'] * 100
            sts14 = result['STS14']['test']['cos_sim']['spearman'] * 100
            sts15 = result['STS15']['test']['cos_sim']['spearman'] * 100
            sts16 = result['STS16']['test']['cos_sim']['spearman'] * 100
            sts17_en = result['STS17']['test']['en-en']['cos_sim']['spearman'] * 100
            sts22_en = result['STS22']['test']['en']['cos_sim']['spearman'] * 100
            sick_r = result['SICK-R']['test']['cos_sim']['spearman'] * 100
            biosses = result['BIOSSES']['test']['cos_sim']['spearman'] * 100

            atec = result['ATEC']['test']['cos_sim']['spearman'] * 100
            lcqmc = result['LCQMC']['test']['cos_sim']['spearman'] * 100
            bq = result['BQ']['test']['cos_sim']['spearman'] * 100
            pawsx = result['PAWSX']['test']['cos_sim']['spearman'] * 100
            qbqtc = result['QBQTC']['test']['cos_sim']['spearman'] * 100
            stsb_zh = result['STSB']['test']['cos_sim']['spearman'] * 100
            sts22_zh = result['STS22']['test']['zh']['cos_sim']['spearman'] * 100
            afqmc = result['AFQMC']['validation']['cos_sim']['spearman'] * 100

            mean_zh = (atec + lcqmc + bq + pawsx + qbqtc + stsb_zh + afqmc + sts22_zh) / 8
            mean_en = (sts12 + sts13 + sts14 + sts15 + sts16 + stsb + sick_r + sts17_en + sts22_en + biosses) / 10

            tb = PrettyTable(["Tatoeba", "BUCC", "Avg."])
            tb.add_row([tatoeba, bucc, (tatoeba+bucc)/2])
            tb.float_format = "4.2"
            print(tb)
            print('')

            tb = PrettyTable(["STS12", "STS13", "STS14", "STS15", "STS16", "STS17",  "STS22", "BIOSSES", "STSB", "SICK-R", "Avg.en"])
            tb.add_row([sts12, sts13, sts14, sts15, sts16, sts17_en, sts22_en, biosses, stsb, sick_r, mean_en])
            tb.float_format = "4.2"
            print(tb)
            print('')

            tb = PrettyTable(["ATEC", "LCQMC", "BQ", "PAWSX", "QBQTC", "AFQMC", "STS-Z", "STS22", "Avg.zh"])
            tb.add_row([atec, lcqmc, bq, pawsx, qbqtc, afqmc, stsb_zh, sts22_zh, mean_zh])
            tb.float_format = "4.2"
            print(tb)
        print_result()


        for k, v in result.items():
            if 'test' in v.keys():
                print(k, v['test'])
            else:
                print(k, v['validation'])

    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 300 == 0:
                if step:
                    eval_now(step)
                else:
                    test_now()

            inputs, labels = batch
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            out_s = model(inputs)

            loss, list_loss = calc_loss(out_s.last_hidden_state[:, 0], labels, device=device, step=step, model=model)
            loss.backward()
            if step % 100 == 0:
                print("epoch:{}, batch:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss), end='     ')
                print('list_loss: ', *[round(item, 8) for item in list_loss], sep='   ')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        test_now()

    model.load_state_dict(torch.load('result/T2S.bin'))
    print('--------------   bast result --------------------')
    test_now()


if __name__ == '__main__':
    train()

















