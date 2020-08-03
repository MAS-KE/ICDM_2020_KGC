# 消费者事件原因抽取基线模型

# Consumer Event Cause Extraction Baseline Model 

​	消费者事件原因提取基线模型任务旨在从给定品牌或产品的文本中提取具有预定义事件类型的消费者事件以及提取事件的原因，属于联合模型。该模型基于预训练模型BERT并采用类似抽取式机器阅读理解(MRC)的模型结构，但与MRC不同的是：

​	The CECE task aims to extract consumer events and the cause of the event from the text of a given brand or product. It belongs to a joint model. This model is based on the pre-trained model BERT and uses a model structure similar to extractive machine reading comprehension (MRC), but it is different from MRC: 

1. 该任务增加了对多种事件类型的判断，这在一般的MRC任务中是没有的。
2. 该任务存在"多个答案片段"也即是事件的多个原因，而一般的MRC任务中一般只选择一个正确答案。

	1. This task adds judgments on multiple event types, which are not found in general MRC tasks. 
 	2.  There are "multiple answer spans" in this task, which means that there are multiple reasons for the event. In general MRC tasks, only one correct answer is generally selected.


### 快速开始：

### Quick Start:

#### 前期准备：

#### Preliminary preparation：

1. python适用版本 3.5以上 (2.7.x版本没有尝试)

   python version 3.5.x or higher (not tried for 2.7.x version)

2. pytorch 0.4版本以上

   pytorch version 0.4 or higher

3. pytorch_transformers==1.1.0 (huggingface)

4. bert_uncased预训练模型,需包含三个文件(pytorch_model.bin,  config.json, vocab.txt)

   The bert_uncased pre-training model needs to contain three files (pytorch_model.bin, config.json, vocab.txt)

5. nltk

6. tqdm



#### 详细步骤：

#### Detail Steps:

步骤1： 准备bert预训练模型，可至 https://github.com/huggingface/transformers 下载。

Step 1:  Prepare the bert pre-training model, which can be downloaded at https://github.com/huggingface/transformers.

步骤2：准备训练数据(train.json)和验证数据(valid.json)，可根据自己的方式对训练数据进行划分，然后分别命名为train.json , dev.json。然后将以上两个文件和valid,json 一起放在raw_data/目录下即可(如果没有raw_data/目录，请自己创建)

Step 2:  Prepare training data (train.json) and validation data (valid.json). You can divide the training data in your own way, and then rename them as train.json and dev.json respectively. Then put the above two files and valid.json together in the raw_data/ directory (if there is no raw_data/ directory, please create it yourself)

步骤3：如果根目录下没有saved_models/ 目录，请自行创建，该目录在训练过程中自动保存最好的checkpoint

Step 3:  If there is no saved_models/ directory in the root directory, please create it yourself. This directory will automatically save the best checkpoint during the training process

步骤4： 注意配置好conf/config.json中的数据路径以及bert_model_path的路径

Step 4: Please configure the data path in conf/config.json and the path of bert_model_path correctly

步骤5：在data/event_query.json文件中记录的是根据事件类型和产品/品牌定义的query, 参赛选手也可以根据自己的意愿重新设定。

Step 5: What is recorded in the data/event_query.json file is the query defined by the event type and product/brand, and contestants can also reset it according to their wishes.

步骤6：根目录下的main.py为程序入口文件，只需要执行 python main.py  即可开始运行。默认进行训练，在该文件末尾有一个"mode"参数，默认为“train”,表示训练；如果为"eval_dev"则基于已保存的checkpoint对验证数据进行指标计算，并返回计算结果。

Step 6: The " main.py"  in the root directory is the program entry file. You only need to execute " python main.py "  to start running and training is performed by default. At the end of the file, there is a "mode" parameter, the default is "train", which means training; if it is "eval_dev", the verification data(dev.json) is calculated based on the saved checkpoint and the calculation result is returned.

步骤7：训练结束后，在根目录下执行:  python reason_extract.py   即可产生要提交的测试数据的结果文件pred_result.json  。

Step 7: After training, execute in the root directory: " python reason_extract.py " to generate the result file " pred_result.json"  of the test data to be submitted.

**注意：对于以上提到的数据以及预训练模型的位置，请务必参考配置文件 config.json。**

**Note: For the data mentioned above and the location of the pre-trained model, please refer to the configuration file" config.json " .**