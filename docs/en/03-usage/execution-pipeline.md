# Execution pipeline summary

It happens in one line, `model(inDict)`, in other words, inside [`NeuralNet::operator()`](neural-net.md#execution).

```text hl_lines="7 8"
model(inDict)
  1. Input validation and assignment
  2. Waiting for previous forward               syncPrevForward()
  3. Model hook                                 onForwardBegin()
  4. Input upload                               toPinned()
  5. Re-recording assessment
  6. Graph preparation (once, initially)        initialize()       ┐
  7. Command buffer recording (if necessary)    recordCommands()   ┘ not run on every call
  8. Submit (Asynchronous)
  9. Return output
```

![Timeline of three consecutive model() calls across the user, NeuralNet, and GPU lanes. Only call 1 runs steps 6 and 7; the GPU keeps running after the call returns; the next call waits for the previous forward to finish](../../assets/execution-timeline.svg)

The key is that **steps 6 and 7 are not executed every time**. As long as the same shape and buffer are maintained, the pre-recorded command buffer is submitted as is. This is why vAi does not go through the graph again for every inference.

### 1. Input validation and assignment

Confirm that the keys of the passed [`TensorDict`](neural-net.md#execution) match the model's input node names exactly, and assign each tensor to the input node's slot.

### 2. Waiting for previous forward

As forward is executed asynchronously, the previous call might still be running on the GPU. Before reusing the host-visible buffer in this call, wait for the previous forward to complete via [`syncPrevForward()`](neural-net.md#when-modifying-the-same-value-outside-the-execution).

### 3. Model hook

Call [`NeuralNet::onForwardBegin()`](neural-net.md#onforwardbegin). It is an override hook that is used when the model has a state that needs to be updated for every forward. For example, the Llama 3 sample updates the KV cache's token counter and sampling random number here.

### 4. Input upload

The input tensor in the [host data](tensor.md#state) state is copied to the pinned buffer (Host-Visible Device Buffer). As the GPU reads this buffer directly, there is no need to transfer it to VRAM again.

Like the output of another forward, an input that already has a device buffer is used as is without a separate copy (zero-copy).

### 5. Re-recording assessment

Confirm whether the shape and buffer of the input tensor are the same as those used when recording the current command buffer. If they are the same, skip the command buffer recording in step 7 entirely.

If you continuously reuse the same input tensor, re-recording will not occur
(see [External input in Tensor](tensor.md#2-external-input)).

### 6. Graph preparation (once, initially)

Executed only on the first forward. Topologically sort the graph and determine the node execution order. If there is a cycle, an exception is thrown. Then verify each node's shape with [`Node::prepare()`](node.md#nodeprepare-and-noderun) and upload the tensor in the [`persistent`](node.md#persistent) slot containing weights and cache to the device buffer.

**The host data of the weights that have finished uploading is released at this time**. There won't be a situation where the same weights remain on both the host and device.

### 7. Command buffer recording (if necessary)

Go through the nodes in topologically sorted order and record the command buffer. Proceed in the following order for each node.

* Verify the input shape with `prepare()` and create the output tensor.
* Assign device buffers from the buffer pool to the tensors in the [`output`](node.md#output) and [`temporary`](node.md#temporary) slots, and share the `output` tensor with the [`input`](node.md#input) slot of the next connected node.
* [`Node::run(cmdBuff)`](node.md#nodeprepare-and-noderun) **records** the commands to be executed by the GPU in the command buffer. But the GPU operation is not executed at this point.

Non-persistent tensors of nodes that have finished recording are cleared immediately, and the buffers they were using are returned to the pool. As the following node may reuse this buffer, intermediate tensors with non-overlapping lifetimes share the same memory.

### 8. Submit

The completed command buffer, in other words, the list of commands that records the GPU's tasks in order, is submitted to the GPU operation queue. **The `model(...)` call returns immediately after submission**, and the GPU continues the tasks asynchronously. Users can continue with the next task without waiting for the GPU operation to finish.

Use a fence to track whether the GPU operation has been completed. The fence set with the submission is signaled when the GPU completes the execution of the command buffer. When [`Tensor::downloadToHost()`](tensor.md#reading-output) is needed to read the result, wait for this fence.

This submission reuses the output buffer that the previous forward was using. If you haven't read the previous output yet, the new forward overwrites that buffer, causing the previous value to be lost.

### 9. Return output

Put the tensor that points to the output boundary buffer into `TensorDict` and return it. At this point, the GPU might still be calculating.

The returned tensor is **a handle that points to the buffer where the result will be recorded, rather than the result itself**. When using the actual values on the host, if you call `Tensor::downloadToHost()`, it waits for the fence when necessary and retrieves the data.
(see [Reading output in Tensor](tensor.md#reading-output)).
