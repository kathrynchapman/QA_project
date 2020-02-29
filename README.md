# QA_project

Rewritten scripts:
  1. pt_cqa_supports.py
      python pt_cqa_supports.py
  2. pt_cqa_gen_batches.py
  
To-do:
- cqa_model.py
	- bert_rep 					(1) READY
	- bert_segment_rep 				(1) No need to re-implement, this function is not used
	- cqa_model 					(0) READY
	- aux_cqa_model 				(0) No need to re-implement, this function is exactly the same as cqa_model
	- yesno_model 					(0) READY
	- followup_model 				(0) No need to re-implement, this function is exactly the same as yesno_model
	- history_attention_net 			(1) IN PROGRESS
	- disable_history_attention_net 		(0) NEXT, almost the same as history_attention_net
	- fine_grained_history_attention_net 		(1)


to run main script from within the 'script' directory: (29.02.2020) <br>
python pt_cqa_run_his_atten.py --do_train --quac_data_dir ../data/ --cache_dir ../data/practice --load_small_portion --do_predict --output_dir ../exps_dir

Note for Kathryn: Did you run the original code? It looks a bit messy in some parts, as if they tried different functions and versions and forgot
to clean it up before setting the definite github repository. I started to wonder if it works at all...

We could define the flags that we want to be flexible as parameters in the command line (with args parse) and the ones that we want to be fixed
at the beginning of the main script
