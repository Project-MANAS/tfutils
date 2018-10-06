## tfutils

tfutils is a Python package written to make using TensorFlow for multi-GPU easier.

For using multiple GPUs, TensorFlow usually expects the user to code using the tf.nn modules, 
however, these modules are not quite as easy to use as compared to some of the higher level APIs. 
For example, in tf.nn, you need to manually calculate the size of the weight matrices required in 
conv operations.

On the other hand, higher level APIs offer the ease of use, but using them to build custom graphs
spanning multiple GPUs is a f***ing pain.

tfutils plans to majorly solve this issue with its `layers` module, by using two extra 
params: `wt_device` and `op_device`.

Apart from this, tfutils also has two more modules: `vector` and `meta`

The `vector` module is used for some convenient operations that aren't supported by TensorFlow out-of-the-box.

The `meta` module is used for dealing with TensorFlow's (kinda complicated) scoping and session systems.

Development is still underway, but most of the important features are already added. Stay tuned.