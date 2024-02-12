import json
import logging
import os
import pathlib
import traceback
from datetime import datetime
from time import time

import datasets
from rich.console import Console
import torch
from mteb import MTEB
from .my_mteb_class import *

logger = logging.getLogger(__name__)

class MyMTEB(MTEB):
    def __init__(self):
        super(MyMTEB, self).__init__()

    def run(
            self,
            model,
            verbosity=1,
            output_folder="results/result",
            eval_splits=None,
            overwrite_results=False,
            raise_error: bool = True,
            **kwargs
    ):
        """
        Run the evaluation pipeline on the selected tasks.

        Parameters
        ----------
        model:
            Model to be used for evaluation
        verbosity: int
            Verbosity level. Default is 1.
            0: print tasks tqdm progress bar
            1: print tasks tqdm progress bar and scores
            2: print everything (including datasets loading)
        output_folder: str
            Folder where the results will be saved
        raise_error: bool
            Whether to raise an error if an exception occurs during evaluation.
        :return: Returns a dictionary of task names and corresponding metrics results.
        """
        # Set logging
        if verbosity < 2:
            datasets.logging.set_verbosity(40)
            datasets.logging.disable_progress_bar()

        # Create output folder
        if output_folder is not None:
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Run selected tasks
        logger.info(f"\n\n## Evaluating {len(self.tasks)} tasks:")
        self.print_selected_tasks()
        evaluation_results = {}
        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(f"\n\n********************** Evaluating {task.description['name']} **********************")

            # skip evaluation if results folder exists and overwrite_results is False
            if output_folder is not None:
                save_path = os.path.join(output_folder, f"{task.description['name']}{task.save_suffix}.json")
                if os.path.exists(save_path) and overwrite_results is False:
                    logger.warning(f"WARNING: {task.description['name']} results already exists. Skipping.")
                    del self.tasks[0]
                    continue

            try:
                task_eval_splits = eval_splits if eval_splits is not None else task.description.get("eval_splits", [])

                # load data
                logger.info(f"Loading dataset for {task.description['name']}")
                # ['test']
                task.load_data(eval_splits=task_eval_splits)

                # run evaluation
                task_results = {
                    # "mteb_version": __version__,
                    "dataset_revision": task.description.get("revision", None),
                    "mteb_dataset_name": task.description["name"],
                }
                for split in task_eval_splits:
                    tick = time()
                    results = task.evaluate(model, split, **kwargs)
                    tock = time()
                    logger.info(f"Evaluation for {task.description['name']} on {split} took {tock - tick:.2f} seconds")
                    results["evaluation_time"] = round(tock - tick, 2)
                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {results}")

                # todo

                # save results
                if output_folder is not None:
                    with open(save_path, "w") as f_out:
                        json.dump(task_results, f_out, indent=2, sort_keys=True)

                evaluation_results[task.description["name"]] = task_results

            except Exception as e:
                logger.error(f"Error while evaluating {task.description['name']}: {e}")
                if raise_error:
                    raise e
                logger.error(f"Please check all the error logs at: {self.err_logs_path}")
                with open(self.err_logs_path, "a") as f_out:
                    f_out.write(f"{datetime.now()} >>> {task.description['name']}\n")
                    f_out.write(traceback.format_exc())
                    f_out.write("\n\n")

            # empty memory
            del self.tasks[0]

        return evaluation_results

def test(model, args):
    model.eval()
    with torch.no_grad():
        evaluation = MTEB(tasks=[MyAFQMC(), MyATEC(), MyBQ(), MyLCQMC(), MyPAWSX(),
                                 MyQBQTC(), MyZSTSB(), MySTS12(), MySTS13(), MySTS14(), MySTS15(), MySTS16(),
                                 MySTSB(), MySickR(), MyTatoeba(langs=["cmn-eng"]), MyBucc(langs=["zh-en"])])
        result = evaluation.run(model, output_folder=f"result/{'student'}", overwrite_results=True, verbosity=1,
                       batch_size=args.batch_size)
    model.train()



