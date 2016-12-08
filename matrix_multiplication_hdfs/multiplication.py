import os
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# read
# http://learningtensorflow.com/ReadingFilesBasic/ for reading files 
# http://stackoverflow.com/questions/37063081/how-does-the-tf-scatter-update-work-inside-the-while-loop for while with scatter_update

# run: 
# CLASSPATH=$(/home/felix/Software/hadoop-2.7.3/bin/hdfs classpath --glob) python multiplication.py

# /home/felix/Software/hadoop-2.7.3/bin/hdfs dfs -put /home/felix/EmmaTensorFlow/matrix_multiplication_hdfs/data/data.csv /
# /home/felix/Software/hadoop-2.7.3/bin/hdfs dfs -ls /

#export JAVA_HOME=/usr/lib/jvm/java-8-oracle
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-oracle"

#export HADOOP_HDFS_HOME=/home/felix/Software/hadoop-2.7.3
os.environ["HADOOP_HDFS_HOME"] = "/home/felix/Software/hadoop-2.7.3"

#export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-oracle/jre/lib/amd64/server
os.environ["LD_LIBRARY_PATH"] = os.environ["JAVA_HOME"] + "/jre/lib/amd64/server"

def read_and_decode(filename_queue):
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  
  # Type information and column names based on the decoded CSV.
  record_defaults = [[0.0], [0.0], [0.0]]
  f1,f2,f3 = tf.decode_csv(value, record_defaults=record_defaults)

  return [f1,f2,f3]

def cond(sequence_len, step):
    return tf.less(step,sequence_len)

def body(sequence_len, step, filename_queue): 
    begin = tf.get_variable("begin",tensor_shape.TensorShape([3, 3]),dtype=tf.float32,initializer=tf.constant_initializer(0))
    begin = tf.scatter_update(begin, step, read_and_decode(filename_queue), use_locking=None)
    tf.get_variable_scope().reuse_variables()

    with tf.control_dependencies([begin]):
        return (sequence_len, step+1)

def get_all_records(FILE):
 with tf.Session() as sess:

    filename_queue = tf.train.string_input_producer([FILE], num_epochs=1, shuffle=False)

    b = lambda sl, st: body(sl,st,filename_queue)

    step = tf.constant(0)
    sequence_len  = tf.constant(3)
    _,step, = tf.while_loop(cond,
                    b,
                    [sequence_len, step], 
                    parallel_iterations=10, 
                    back_prop=True, 
                    swap_memory=False, 
                    name=None)

    begin = tf.get_variable("begin",tensor_shape.TensorShape([3, 3]),dtype=tf.float32)

    with tf.control_dependencies([step]): #wait for the loop to finish
      product = tf.matmul(begin, begin)

    init0 = tf.local_variables_initializer()
    sess.run(init0)
    init1 = tf.global_variables_initializer()
    sess.run(init1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      print(sess.run([product]))
    except tf.errors.OutOfRangeError, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
   
    coord.join(threads)
   
get_all_records('hdfs://default/data.csv')
