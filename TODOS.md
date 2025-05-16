# Faster Training
- [x] Implement static batch size on both hint_steps and num_nodes for each algo dataset
- [x] Fix evaluator to take the num_steps and apply the mask to the hints
- [x] Set up passing static batch to the model
- [ ] Ament feature to contain
- [ ] Test that adjacency matrix is correctly computed on static batches
- [ ] Pass the num nodes to the model
- [ ] Ensure that the output for static batch matches the non-static batch

# Evaluation
- [ ] Implement output accuracy/score separate from hint

# Monitoring
- [ ] Remove steps done and examples seem from per algo monitoring
- [ ] Make x-axis the total steps