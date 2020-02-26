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
	- followup_model 				(0) READY, same as yesno_model with another name
	- history_attention_net 			(1)
	- disable_history_attention_net 		(0)
	- fine_grained_history_attention_net 		(1)


Note for Kathryn: Did you run the original code? It looks a bit messy in some parts, as if they tried different functions and versions and forgot
to clean it up before setting the definite github repository. I started to wonder if it works at all...
