#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月16日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(p)
import tensorflow as tf
import numpy as np
import jieba
from constant import PROJECT_DIRECTORY, sogou_classification_label_list
from data.prepare import get_sogou_classification_stopwords_set
from utils.utils import ensure_unicode
from word2vec.data_convert import get_text_converter_for_sogou_classification


# checkpoint_dir, 训练时保存的模型
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(PROJECT_DIRECTORY, "classification/rnn/lstm/data/model/runs/1495005375/checkpoints"), "Checkpoint directory from training run")
# max_sentence_length, 文本最大长度
tf.flags.DEFINE_integer("max_sentence_length", 500, "max sentence length")
# allow_soft_placement, 设置为True时, 如果你指定的设备不存在，允许TF自动分配设备
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# log_device_placement, 设备上放置操作日志的位置
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 设置参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def predict_doc(text):
    """
    给定一个文本,预测文本的分类
    """
    text = ensure_unicode(text)
    stopwords_set = get_sogou_classification_stopwords_set()
    segment_list = jieba.cut(text)
    word_list = []
    for word in segment_list:
        word = word.strip()
        if '' != word and word not in stopwords_set:
            word_list.append(word)
    word_segment = ' '.join(word_list)

    # 查找最新保存的检查点文件的文件名
    checkpoint_dir = FLAGS.checkpoint_dir
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    index2label_dict = {i: l.strip() for i, l in enumerate(sogou_classification_label_list)}
    converter = get_text_converter_for_sogou_classification(FLAGS.max_sentence_length)
    x_test = []
    for doc, _ in converter.transform_to_ids([word_segment]):
        x_test.append(doc)
    x_test = np.array(x_test)
    with tf.Graph().as_default() as graph:
        # session配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        # 自定义session然后通过session.as_default() 设置为默认视图
        with tf.Session(config=session_conf).as_default() as sess:
            # 载入保存Meta图
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # 恢复变量
            saver.restore(sess, checkpoint_file)
            # 从图中根据名称获取占位符
            input_x = graph.get_operation_by_name("model/input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("model/dropout_keep_prob").outputs[0]

            # 待评估的Tensors
            prediction = graph.get_operation_by_name("model/output/prediction").outputs[0]
            predict_class = sess.run(prediction, {input_x: x_test, dropout_keep_prob: 1.0})[0]
            return index2label_dict.get(predict_class)


if __name__ == '__main__':
    print (predict_doc("这是一次跨越世纪的重逢，这是一次弥补历史的棋坛盛事。5月23日至27日，在美丽的沈阳世博园，来自中国、日本和韩国三国的九位当年叱咤世界棋坛的围棋元老，将上演一次对决。昨天，本次元老赛的参赛者之一、中国围棋协会主席陈祖德表示，他对即将在沈阳揭幕的2006年中、日、韩三国围棋元老赛非常期待：“这是一次弥补历史的比赛！”　　三国元老首次聚首　　本次三国元老赛由国家体育总局棋牌运动管理中心、中国围棋协会主办、由沈阳市体育局、沈阳市体育总会、沈阳晚报、沈阳市围棋协会、沈阳电视台协办。比赛地点设在沈阳世界园艺博览会，是2006年沈阳市承办的级别最高的比赛，也是2006年国内外广泛关注的重要体育赛事之一。　　参加比赛的中日韩围棋元老都是二十世纪世界最著名的围棋代表人物，他们分别是中国队的陈祖德、王汝南和聂卫平；日本队的林海峰、宫本直毅和羽根泰正；韩国队的金寅、河灿锡和尹琦铉。除了中国的三位元老赫赫有名外，日本的林海峰是一代围棋巨人吴清源的弟子，曾三连霸名人战，五获本因坊，被日本《棋道》杂志敬称为棋界“阿信”；宫本直毅九段师从于关西棋院创始人桥本宇太郎九段，他在1974年率团访华，对聂卫平一生的命运发挥过至关重要的影响。中国棋迷相对陌生的金寅更是韩国著名的超一流九段棋手。相当于韩国的“陈祖德”，从1965年开始，韩国进入了名副其实的“金寅时代”，曾影响韩国围棋界十多年，七十年代后期，由于曹薰铉、徐奉洙等新人的崛起，金寅渐渐地退出了棋战的第一线。但正是他将韩国围棋引上了现代之路。陈祖德：这是弥补历史的比赛　　中国围棋协会主席、原中国棋院院长陈祖德先生，昨天在接受本报记者采访时对这次比赛给予了很高的评价，“这是世界第一次！”陈老说，“以前我们只是分别搞过中日和中韩的元老比赛，我们这批三个国家的棋手聚会还是第一次，因此这次比赛让人感到很兴奋。”　　陈祖德介绍说，因为各种历史原因，他和那个时代的韩日两国的高手们在各自棋力达到顶峰的时候没有交过手，这可以算是一个历史的遗憾，而这次三国元老赛正是弥补了他那个时代的选手的一个遗憾，填补世界围棋的一次历史空白，同时也将对中国围棋产生深远的影响。　　当陈祖德听说比赛要在美丽的沈阳世博园举行时，显得非常的高兴：“那太好了，我早就听说那里非常美，我想在那里比赛将是一次享受！这次聚会太令人期待了！”　　世博园将书写世界围棋佳话　　美丽的沈阳世博园已迎来八方游客，而这次九位中日韩围棋元老的跨世纪聚会，将为这美丽的地方书写一段佳话。据了解，本次比赛将在25和26日两天举行交叉比赛，到时候，当年未能一决雌雄的围棋元老们将亮出各自的绝活，留下一个个经典的对局。　　棋盘山，流传着传说与故事，凝聚着历史与现实融合的美丽，这里，将迎来一代宗师们的笑语；世博园，汇聚着鲜花与绿草，传递着人与自然和谐共生的理念，这里，还将留下世界围棋的一段佳话……"))
