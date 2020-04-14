# QA_project

Rewritten scripts:
  1. pt_cqa_supports.py
      python pt_cqa_supports.py
  2. pt_cqa_gen_batches.py
  3. bert_model.py
  4. cqa_model.py
	- bert_rep (updated) 				[READY] <br>
	- bert_segment_rep 				[No need to re-implement, this function is not used] <br>
	- cqa_model 					[READY]<br>
	- aux_cqa_model 				[No need to re-implement, this function is exactly the same as cqa_model]<br>
	- yesno_model 					[READY]<br>
	- followup_model 				[No need to re-implement, this function is exactly the same as yesno_model]<br>
	- history_attention_net 			[READY]<br>
	- disable_history_attention_net 		[READY]<br>
	- fine_grained_history_attention_net 		[READY]<br>


To run main script from within the 'script' directory: (14.04.2020) <br>
python main_qa_script.py --do_train --quac_data_dir ../data/ --cache_dir ../data/practice --do_predict --output_dir ../exps_dir --overwrite_output_dir --batch_size 24
