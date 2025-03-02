import torch
from random import shuffle
from tqdm import tqdm
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_anticipation import AnticipationModel
from model_phase import PhaseModel
import util_train as util
from corruptions import corruption
import time
from ipdb import set_trace

opts = parser.parse_args()

if not opts.random_seed:
	torch.manual_seed(7)

if opts.task == 'anticipation':
	model = AnticipationModel(opts)
elif opts.task == 'phase':
	model = PhaseModel(opts)

if opts.only_temporal:
	train_set, val_set, test_set = prepare_image_features(model.net, opts)
else:
	print(f'train.py is calling this function: prepare_dataset- line 25')
	ds_start_time = time.time()
	train_set, val_set, test_set = prepare_dataset(opts)
	ds_end_time = time.time()
	print(f"Dataset preparation time: {ds_end_time - ds_start_time:.4f} seconds")

with open(model.log_path, "w") as log_file:
	log_file.write(f'{model}\n')
	log_file.flush()

	start_epoch = util.get_start_epoch(opts)
	num_iters_per_epoch = util.get_iters_per_epoch(train_set, opts)
	print('add corrupions in train.py line 60')
	print('corruption:', opts.corruption)

	for epoch in range(start_epoch, opts.epochs + 1):
		# Overall epoch timer
		epoch_start_time = time.perf_counter()

		# Timers for training components (measured only within the training loop)
		train_data_loading_time = 0
		train_forward_time = 0
		train_loss_time = 0
		train_update_weights_time = 0
		train_update_stats_time = 0

		# Timer for training loop overall
		train_loop_start = time.perf_counter()

		model.reset_stats()
		model.net.train()
		if opts.bn_off:
			model.net.cnn.eval()

		mainloopi = 0
		# Training Loop
		for _, op in tqdm(train_set, desc=f"Epoch {epoch} Training"):
			if not opts.image_based:
				model.net.temporal_head.reset()
			
			print(f"Main loop i: {mainloopi}")
			mainloopi += 1

			model.metric_meter['train'].start_new_op()  # necessary to compute video-wise metrics

			for i, (data, target) in enumerate(op):
				
				print(f"Sub loop i: {i}")
				if not opts.image_based and opts.shuffle:
					model.net.temporal_head.reset()

				# --- Data Loading Timing (prepare_batch and corruption) ---
				t1 = time.perf_counter()
				data, target = prepare_batch(data, target)
				if opts.corruption is not None:
					data = corruption(data, opts.corruption)
				t2 = time.perf_counter()
				train_data_loading_time += (t2 - t1)

				# --- Forward Pass Timing ---
				t3 = time.perf_counter()
				output = model.forward(data)
				t4 = time.perf_counter()
				train_forward_time += (t4 - t3)

				# --- Loss Computation Timing ---
				t5 = time.perf_counter()
				loss = model.compute_loss(output, target)
				t6 = time.perf_counter()
				train_loss_time += (t6 - t5)

				# --- Weights Update Timing ---
				t7 = time.perf_counter()
				model.update_weights(loss)
				t8 = time.perf_counter()
				train_update_weights_time += (t8 - t7)

				# --- Stats Update Timing ---
				t9 = time.perf_counter()
				# model.update_stats(
				# 	loss.item(),
				# 	output,
				# 	target,
				# 	mode='train'
				# )
				t10 = time.perf_counter()
				train_update_stats_time += (t10 - t9)

				if opts.shuffle and (i + 1) >= num_iters_per_epoch:
					break

		train_loop_end = time.perf_counter()
		train_loop_time = train_loop_end - train_loop_start

		# Compute "overhead" time in training loop: time not accounted for in measured sub-components.
		measured_train_time = (train_data_loading_time + train_forward_time +
							   train_loss_time + train_update_weights_time + train_update_stats_time)
		train_overhead = train_loop_time - measured_train_time

		# --- Evaluation Loop Timing ---
		eval_loop_start = time.perf_counter()

		# Timers for evaluation components (combined for both val and test)
		eval_data_loading_time = 0
		eval_forward_time = 0
		eval_loss_time = 0

		for mode in ['val', 'test']:
			eval_set = tqdm(val_set, desc=f"Epoch {epoch} {mode.upper()}") if mode == 'val' else test_set

			for _, op in eval_set:
				if not opts.image_based:
					model.net.temporal_head.reset()

				model.metric_meter[mode].start_new_op()

				for data, target in op:
					# --- Data Loading Timing for Eval ---
					et1 = time.perf_counter()
					data, target = prepare_batch(data, target)
					et2 = time.perf_counter()
					eval_data_loading_time += (et2 - et1)

					# --- Forward Pass Timing for Eval ---
					et3 = time.perf_counter()
					if opts.sliding_window:
						output = model.forward_sliding_window(data)
					else:
						output = model.forward(data)
					et4 = time.perf_counter()
					eval_forward_time += (et4 - et3)

					# --- Loss Computation Timing for Eval ---
					et5 = time.perf_counter()
					loss = model.compute_loss(output, target)
					et6 = time.perf_counter()
					eval_loss_time += (et6 - et5)

					# model.update_stats(
					# 	loss.item(),
					# 	output,
					# 	target,
					# 	mode=mode
					# )

		eval_loop_end = time.perf_counter()
		eval_loop_time = eval_loop_end - eval_loop_start

		# Overall epoch time and unmeasured overhead in the epoch (includes training and eval overhead plus any extra)
		epoch_end_time = time.perf_counter()
		epoch_duration = epoch_end_time - epoch_start_time
		measured_total = train_loop_time + eval_loop_time
		epoch_overhead = epoch_duration - measured_total

		# Compute percentages for training components relative to the training loop time
		def perc(part):
			return (part / train_loop_time) * 100 if train_loop_time > 0 else 0

		print(f"\nEpoch {epoch} Timing Summary:")
		print(f"Overall Epoch Time: {epoch_duration:.4f} s")
		print("\n--- Training Loop Breakdown (Total: {0:.4f} s) ---".format(train_loop_time))
		print(f"  Data Loading         : {train_data_loading_time:.4f} s ({perc(train_data_loading_time):.2f}%)")
		print(f"  Forward Pass         : {train_forward_time:.4f} s ({perc(train_forward_time):.2f}%)")
		print(f"  Loss Computation     : {train_loss_time:.4f} s ({perc(train_loss_time):.2f}%)")
		print(f"  Weights Update       : {train_update_weights_time:.4f} s ({perc(train_update_weights_time):.2f}%)")
		print(f"  Stats Update         : {train_update_stats_time:.4f} s ({perc(train_update_stats_time):.2f}%)")
		print(f"  Training Overhead    : {train_overhead:.4f} s ({perc(train_overhead):.2f}%)")
		train_total_measured = (train_data_loading_time + train_forward_time +
								train_loss_time + train_update_weights_time + train_update_stats_time + train_overhead)
		print(f"  Total (Measured)     : {train_total_measured:.4f} s (should equal {train_loop_time:.4f} s)")

		# Compute percentages for evaluation components relative to the evaluation loop time
		def perc_eval(part):
			return (part / eval_loop_time) * 100 if eval_loop_time > 0 else 0

		print("\n--- Evaluation Loop Breakdown (Total: {0:.4f} s) ---".format(eval_loop_time))
		print(f"  Data Loading         : {eval_data_loading_time:.4f} s ({perc_eval(eval_data_loading_time):.2f}%)")
		print(f"  Forward Pass         : {eval_forward_time:.4f} s ({perc_eval(eval_forward_time):.2f}%)")
		print(f"  Loss Computation     : {eval_loss_time:.4f} s ({perc_eval(eval_loss_time):.2f}%)")
		eval_measured_total = eval_data_loading_time + eval_forward_time + eval_loss_time
		print(f"  Total (Measured)     : {eval_measured_total:.4f} s (should be <= {eval_loop_time:.4f} s)")
		print(f"\nUnaccounted Epoch Overhead (outside training/eval loops): {epoch_overhead:.4f} s")

		model.summary(log_file, epoch)
