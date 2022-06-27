import tensorflow as tf
import numpy as np
from modules import *
import math
from tensorflow.python.layers import core as layers_core
from data_utils import *
import random
class TCVAE():
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.vocab_size = hparams.from_vocab_size
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.encoder_num_layers = hparams.encoder_num_layers
        self.decoder_num_layers = hparams.decoder_num_layers
        self.num_heads = hparams.num_heads
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.max_story_length = 210
        self.max_length = 110
        self.max_single_length = 105
        self.latent_dim = hparams.latent_dim
        self.dropout_rate = hparams.dropout_rate
        self.init_weight = hparams.init_weight
        self.flag = True
        self.mode = mode
        self.batch_size = hparams.batch_size
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.input_ids = tf.placeholder(tf.int32, [None, None])
            self.input_scopes = tf.placeholder(tf.int32, [None, None])
            self.input_positions = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.int32, [None, None, None])
            self.input_lens = tf.placeholder(tf.int32, [None])
            self.targets = tf.placeholder(tf.int32, [None, None])
            self.weights = tf.placeholder(tf.float32, [None, None])
            self.input_windows = tf.placeholder(tf.float32, [None, 1, None])
            self.which = tf.placeholder(tf.int32, [None])

        else:
            #如果是评估或者推理则执行这里
            self.input_ids = tf.placeholder(tf.int32, [None, None])
            self.input_scopes = tf.placeholder(tf.int32, [None, None])
            self.input_positions = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.int32, [None, None, None])
            self.input_lens = tf.placeholder(tf.int32, [None])
            self.input_windows = tf.placeholder(tf.float32, [None, 1, None])
            self.which = tf.placeholder(tf.int32, [None])


        with tf.variable_scope("embedding") as scope:
            self.word_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
            # self.word_embeddings = tf.Variable(hparams.embeddings, trainable=True)
            self.scope_embeddings = tf.Variable(self.init_matrix([9, int(self.emb_dim/2)]))

        with tf.variable_scope("project"):
            self.output_layer = layers_core.Dense(self.vocab_size, use_bias=True)
            self.mid_output_layer = layers_core.Dense(self.vocab_size, use_bias=True)
            self.input_layer = layers_core.Dense(self.num_units, use_bias=False)


        with tf.variable_scope("encoder") as scope:
            self.word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_ids)
            self.scope_emb = tf.nn.embedding_lookup(self.scope_embeddings, self.input_scopes)
            self.pos_emb = positional_encoding(self.input_positions, self.batch_size, self.max_length, int(self.emb_dim/2))

            # self.embs = self.word_emb + self.scope_emb + self.pos_emb
            self.embs = tf.concat([self.word_emb, self.scope_emb, self.pos_emb], axis=2)
            inputs = self.input_layer(self.embs)


            self.query = tf.get_variable("w_Q", [1, self.num_units], dtype=tf.float32)
            windows = tf.transpose(self.input_windows, [1, 0, 2])
            layers_outputs = []

            post_inputs = inputs
            for i in range(self.decoder_num_layers):
                with tf.variable_scope("num_layers_{}".format(i)):
                    outputs = multihead_attention(queries=inputs,
                                                  keys=inputs,
                                                  query_length=self.input_lens,
                                                  key_length=self.input_lens,
                                                  num_units=self.num_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  using_mask=True,
                                                  mymasks=self.input_masks,
                                                  scope="self_attention")

                    outputs = outputs + inputs
                    inputs = normalize(outputs)

                    outputs = feedforward(inputs, [self.num_units * 2, self.num_units], is_training=self.is_training, dropout_rate=self.dropout_rate, scope="f1")
                    outputs = outputs + inputs
                    inputs = normalize(outputs)

            for i in range(self.encoder_num_layers):
                with tf.variable_scope("num_layers_{}".format(i)):
                    post_outputs = multihead_attention(queries=post_inputs,
                                                  keys=post_inputs,
                                                  query_length=self.input_lens,
                                                  key_length=self.input_lens,
                                                  num_units=self.num_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  using_mask=False,
                                                  mymasks=None,
                                                  scope="self_attention",
                                                  reuse=tf.AUTO_REUSE
                                                 )


                    post_outputs = post_outputs + post_inputs
                    post_inputs = normalize(post_outputs)

                    post_outputs = feedforward(post_inputs, [self.num_units * 2, self.num_units], is_training=self.is_training,
                                          dropout_rate=self.dropout_rate, scope="f1",reuse=tf.AUTO_REUSE)
                    post_outputs = post_outputs + post_inputs
                    post_inputs = normalize(post_outputs)

            big_window = windows[0]
            post_encode, weight = w_encoder_attention(self.query,
                                                 post_inputs,
                                                 self.input_lens,
                                                 num_units=self.num_units,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 is_training=self.is_training,
                                                 using_mask=False,
                                                 mymasks=None,
                                                 scope="concentrate_attention"
                                                 )

            prior_encode, weight = w_encoder_attention(self.query,
                                                      inputs,
                                                      self.input_lens,
                                                      num_units=self.num_units,
                                                      num_heads=self.num_heads,
                                                      dropout_rate=self.dropout_rate,
                                                      is_training=self.is_training,
                                                      using_mask=True,
                                                      mymasks=big_window,
                                                      scope="concentrate_attention",
                                                       reuse=tf.AUTO_REUSE
                                                      )

            post_mulogvar = tf.layers.dense(post_encode, self.latent_dim * 2, use_bias=False, name="post_fc")
            post_mu, post_logvar = tf.split(post_mulogvar, 2, axis=1)

            prior_mulogvar = tf.layers.dense(tf.layers.dense(prior_encode, 256, activation=tf.nn.tanh), self.latent_dim * 2, use_bias=False, name="prior_fc")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)


            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                latent_sample = sample_gaussian(post_mu, post_logvar)
            else:
                latent_sample = sample_gaussian(prior_mu, prior_logvar)


            self.latent_sample = latent_sample
            latent_sample = tf.tile(tf.expand_dims(latent_sample, 1), [1, self.max_story_length, 1])
            inputs = tf.concat([inputs, latent_sample], axis=2)
            inputs = tf.layers.dense(inputs, self.num_units, activation=tf.tanh, use_bias=False, name="last")


            self.logits = self.output_layer(inputs)
            self.s = self.logits
            self.sample_id = tf.argmax(self.logits, axis=2)
            # self.sample_id = tf.argmax(self.weight_probs, axis=2)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.variable_scope("loss") as scope:
                self.global_step = tf.Variable(0, trainable=False)
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)

                self.total_loss = tf.reduce_sum(crossent * self.weights)

                kl_weights = tf.minimum(tf.to_float(self.global_step) / 20000, 1.0)
                kld = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
                self.loss = tf.reduce_mean(crossent * self.weights) + tf.reduce_mean(kld) * kl_weights


        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.variable_scope("train_op") as scope:
                optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.9, beta2=0.99, epsilon=1e-9)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=120)

    #处理输入的数据
    def get_batch(self, data, no_random=False, id=0, which=0):
        hparams = self.hparams
        input_scopes = []
        input_ids = []
        input_positions = []
        input_lens = []
        input_masks = []
        input_which = []
        input_windows = []
        targets = []
        weights = []
        for i in range(hparams.batch_size):
            if no_random:
                x = data[(id + i) % len(data)]
                #which_stn = (id + i) % 5
                which_stn = 1
                # which_stn = which
            else:
                x = random.choice(data)
                which_stn = 1
                #which_stn = random.randint(0, 4)

            input_which.append(which_stn)
            mask = []
            input_scope = []
            input_id = []
            input_position = []
            input_mask = []
            target = []
            weight = []
            for j in range(0,2):
                input_id.append(GO_ID)
                input_scope.append(j)
                input_position.append(0)
                for k in range(0, len(x[j])):
                    input_id.append(x[j][k])
                    input_scope.append(j)
                    input_position.append(k + 1)
                    target.append(x[j][k])
                    if j == which_stn:
                        weight.append(1.0)
                        mask.append(0)
                    else:
                        weight.append(0.0)
                        mask.append(1)
                target.append(EOS_ID)
                if j == which_stn:
                    weight.append(1.0)
                    mask.append(0)
                else:
                    weight.append(0.0)
                    mask.append(1)
                input_id.append(EOS_ID)
                input_scope.append(j)
                input_position.append(len(x[j]) + 1)
                target.append(GO_ID)
                if j == which_stn:
                    weight.append(0.0)
                    mask.append(0)
                else:
                    weight.append(0.0)
                    mask.append(1)
                if j == which_stn:
                    for k in range(len(x[j]) + 2, self.max_single_length):
                        input_id.append(PAD_ID)
                        input_scope.append(j)
                        input_position.append(k)
                        target.append(PAD_ID)
                        weight.append(0.0)
                        mask.append(0)
            input_lens.append(len(input_id))
            for k in range(0, self.max_story_length - input_lens[i]):
                input_id.append(PAD_ID)
                input_scope.append(1)
                input_position.append(0)
                target.append(PAD_ID)
                weight.append(0.0)
                mask.append(0)


            input_ids.append(input_id)
            input_scopes.append(input_scope)
            input_positions.append(input_position)
            targets.append(target)
            weights.append(weight)

            tmp_mask = mask.copy()
            last = 0
            window = []

            for k in range(0,2):
                start = last
                if k != 1:
                    last = input_scope.index(k + 1)
                else:
                    last = self.max_story_length
                if k != which_stn:
                    window.append([0] * start + [1] * (last - start) + [0] * (self.max_story_length - last))
            input_windows.append(window)

            for k in range(input_lens[i]):
                if input_scope[k] != which_stn:

                    input_mask.append(mask)
                else:
                    tmp_mask[k] = 1
                    input_mask.append(tmp_mask.copy())

            for k in range(input_lens[i], self.max_story_length):
                input_mask.append(mask)


            input_mask = np.array(input_mask)
            input_masks.append(input_mask)

        return input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows

    def train_step(self, sess, data):
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(data)
        feed = {
            self.input_ids:input_ids,
            self.input_scopes:input_scopes,
            self.input_positions:input_positions,
            self.input_masks:input_masks,
            self.input_lens:input_lens,
            self.weights:weights,
            self.targets:targets,
            self.input_windows:input_windows,
            self.which: input_which
        }
        word_nums = sum(sum(weight) for weight in weights)
        loss, global_step, _, total_loss = sess.run([self.loss, self.global_step, self.train_op, self.total_loss],
                                                        feed_dict=feed)
        return total_loss, global_step, word_nums

    def eval_step(self, sess, data, no_random=False, id=0):
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(
            data, no_random, id)
        feed = {
            self.input_ids: input_ids,
            self.input_scopes: input_scopes,
            self.input_positions: input_positions,
            self.input_masks: input_masks,
            self.input_lens: input_lens,
            self.weights: weights,
            self.targets: targets,
            self.input_windows: input_windows,
            self.which: input_which
        }
        loss, logits = sess.run([self.total_loss, self.logits],
                                                feed_dict=feed)
        word_nums = sum(sum(weight) for weight in weights)
        return loss, word_nums

    def infer_step(self, sess, data, no_random=False, id=0, which=0):
        tf.enable_eager_execution()
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(
            data, no_random, id, which=which)
        input_ids = np.asarray(input_ids).repeat(self.hparams.beam_size,axis=0).tolist()
        input_scopes = np.asarray(input_scopes).repeat(self.hparams.beam_size,axis=0).tolist()
        input_positions = np.asarray(input_positions).repeat(self.hparams.beam_size,axis=0).tolist()
        input_masks = np.asarray(input_masks).repeat(self.hparams.beam_size,axis=0).tolist()
        input_lens = np.asarray(input_lens).repeat(self.hparams.beam_size,axis=0).tolist()
        input_which = np.asarray(input_which).repeat(self.hparams.beam_size,axis=0).tolist()
        input_windows = np.asarray(input_windows).repeat(self.hparams.beam_size,axis=0).tolist()
        start_pos = []
        given = []
        ans = []
        predict = []
        hparams = self.hparams
        beam_scores_begin = tf.zeros((self.hparams.batch_size, 1), dtype=tf.float32)
        beam_scores_end = tf.ones((self.hparams.batch_size, self.hparams.beam_size - 1), dtype=tf.float32) * (-1e9)# 需要初始化为-inf
        beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)
        beam_scores = tf.reshape(beam_scores, (self.hparams.batch_size * self.hparams.beam_size,))
        done = [False for _ in range(self.hparams.batch_size)] # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.hparams.beam_size, self.max_single_length, length_penalty=0.7)
                for _ in range(self.hparams.batch_size)
        ] # 为每个输入句子定义维护其beam search序列的类实例
        for i in range(self.hparams.batch_size * self.hparams.beam_size):
            start_pos.append(input_scopes[i].index(input_which[i]))
            if i % self.hparams.beam_size == 0:
                given.append(input_ids[i][:start_pos[i]] + [UNK_ID] * self.max_single_length + input_ids[i][start_pos[i] + self.max_single_length:])
                ans.append(input_ids[i][start_pos[i]: start_pos[i]+self.max_single_length].copy())
            predict.append([])

        for i in range(self.max_single_length - 1):
            feed = {
                self.input_ids: input_ids,
                self.input_scopes: input_scopes,
                self.input_positions: input_positions,
                self.input_masks: input_masks,
                self.input_lens: input_lens,
                self.input_windows: input_windows,
                self.which: input_which
            }
            next_token_logits = [None] * self.hparams.batch_size * self.hparams.beam_size
            sample_id, logits = sess.run([self.sample_id, self.logits], feed_dict=feed)
            for j in range(self.hparams.batch_size * self.hparams.beam_size):
                next_token_logits[j] = logits[j, start_pos[j] + i, :] #获取最新一个时刻的预测结果
            scores = tf.nn.log_softmax(next_token_logits, axis=-1) # log_softmax  (batch_size * num_beams, vocab_size)
            next_scores = scores + tf.broadcast_to(beam_scores[:, None],(self.hparams.batch_size * self.hparams.beam_size, self.vocab_size)) # 累加上以前的scores
            next_scores = tf.reshape(next_scores,(self.hparams.batch_size, self.hparams.beam_size * self.vocab_size))
            next_scores, next_tokens = tf.math.top_k(next_scores, k=2 * self.hparams.beam_size, sorted=True)
            next_scores = next_scores.numpy()
            next_tokens = next_tokens.numpy() #[ 20  39  66  30 125  51]
            next_batch_beam = []

            for batch_idx in range(self.hparams.batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, PAD_ID, 0)] * self.hparams.beam_size)  # pad the batch
                    continue
                next_sent_beam = [] # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_size # 1
                    token_id = beam_token_id % self.vocab_size # 1
	    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
	    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
	    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.hparams.beam_size + beam_id
	     # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if (EOS_ID is not None) and (token_id == EOS_ID):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.hparams.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        if i== 0:
                            generated_hyps[batch_idx].add(
                                 input_ids[effective_beam_id][start_pos[effective_beam_id] + 1:start_pos[effective_beam_id] + i + 2], beam_token_score
                            )
                        else:
                            generated_hyps[batch_idx].add(
                                 input_ids[effective_beam_id][start_pos[effective_beam_id] + 1:start_pos[effective_beam_id] + i + 1], beam_token_score
                            )
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    if len(next_sent_beam) == self.hparams.beam_size:
                        break
                # 当前batch是否解码完所有句子
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    max(next_scores[batch_idx]), i
                 )   # 注意这里取当前batch的所有log_prob的最大值
                # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
            # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            # 准备下一次循环(下一层的解码)
            # beam_scores: (num_beams * batch_size)
            # beam_tokens: (num_beams * batch_size)
            # beam_idx: (num_beams * batch_size)
            # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
            # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
            # 生成新的数组
            beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
            beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
            beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面, 
            # 因为有些beam id对应的句子已经解码完了
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
            input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx]).numpy().tolist()
            for batch in range(self.hparams.batch_size * self.hparams.beam_size):
                input_ids[batch][start_pos[batch] + i + 1] = beam_tokens[batch].numpy()
                predict[batch].append(sample_id[batch][start_pos[batch] + i])
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
        # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
        for batch_idx in range(self.hparams.batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.hparams.beam_size):
    	# 对于每个batch_idx的每句beam，都执行加入add
	# 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.hparams.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].numpy().item()
                final_tokens = input_ids[effective_beam_id][start_pos[effective_beam_id]:]
                generated_hyps[batch_idx].add(final_tokens, final_score)
         # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
         # 下面选择若干最好的序列输出
         # 每个样本返回几个句子
        output_num_return_sequences_per_batch = self.hparams.beam_size
        output_batch_size = output_num_return_sequences_per_batch * self.hparams.batch_size
         # 记录每个返回句子的长度，用于后面pad
        sent_lengths_list = []
        best = []
        # 检索最佳的假设
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths_list.append(len(best_hyp))
                best.append(best_hyp)

        sent_lengths = sent_lengths_list

        if min(sent_lengths) != max(sent_lengths):
            sent_max_len = min(max(sent_lengths) + 1, self.max_single_length)
            # 填充PAD
            decoded_list = []

            # 填充内容
            for i, hypo in enumerate(best):
                #如果是最长的句子，就不需要PAD
                if sent_lengths[i] == sent_max_len:
                    decoded_slice = hypo
                else:
                    #将其他句子补添PAD以保证所有句子长度一致
                    num_pad_tokens = sent_max_len - sent_lengths[i]
                    if num_pad_tokens >= 0:
                        padding = PAD_ID * tf.ones((num_pad_tokens,), dtype=tf.int32)
                        decoded_slice = tf.concat([hypo, padding], axis=-1)
                    else:
                        decoded_slice = hypo[:sent_max_len]
                    #使用EOS token完成句子
                    if sent_lengths[i] < self.max_single_length:
                        decoded_slice = tf.where(
                            tf.range(sent_max_len, dtype=tf.int32).numpy() == sent_lengths[i],
                            EOS_ID * tf.ones((sent_max_len,), dtype=tf.int32),
                            decoded_slice,
                        )
                #将解析好的句子加入到列表中
                decoded_list.append(decoded_slice)
            #将列表堆叠起来
            decoded = tf.stack(decoded_list)
        else:
            # 否则直接堆叠起来
            decoded = tf.stack(best)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        tf.disable_eager_execution()
        return given, ans, decoded.numpy().tolist(),predict


    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring GO_ID
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
