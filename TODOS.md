# Faster Training
- [x] Implement static batch size on both hint_steps and num_nodes for each algo dataset
- [x] Fix evaluator to take the num_steps and apply the mask to the hints
- [x] Set up passing static batch to the model
- [x] Ament feature to contain nb_nodes
- [x] Expand dims method
- [x] Encoding works on static_batch!! 
- [x] Refactor steps mask
- [x] Set up encoder comparison
- [x] Test that adjacency matrix is correctly computed on static batches
- [x] Pass the num nodes to the model
- [x] Test all pgn models
- [x] Test all gat models
- [x] Make static batches work with loss
- [x] Make static batches work with lstm


# Next make the evaluation and loss work
- [x] Check if step mask is correct for evaluation

- [ ] Add device to get mask
- [ ] Make decoder work on static batches
- [ ] Ensure that the output for static batch matches the non-static batch

# Evaluation
- [ ] Implement output accuracy/score separate from hint

# Monitoring
- [ ] Remove steps done and examples seem from per algo monitoring
- [ ] Make x-axis the total steps