# QA_project - Attentive History Selection for Conversational Question Answering

Github for original TF code: https://github.com/prdwb/attentive_history_selection
Paper we are implementing in PyTorch: https://arxiv.org/abs/1908.09456

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


To run main script from within the 'script' directory: (17.04.2020) <br>
python main_qa_script.py --quac_data_dir ../data/ --cache_dir ../data/practice --output_dir ../exps_dir --batch_size 24 --do_train --do_eval
