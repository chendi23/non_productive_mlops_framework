import tensorflow as tf
tf = tf.compat.v1

def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)

    with tf.name_scope('inputs'):
        Xi = tf.to_int32(features['Xi'])
        Xv = features['Xv']

    with tf.name_scope('embeddings'):
        Xi_embedding_matrix = tf.Variable(dtype=tf.float32,
                                          initial_value=tf.random_normal(shape=[params.feature_dim, params.emb_dim]),)
        Xi_embeddings = tf.nn.embedding_lookup(Xi_embedding_matrix, Xi)
        Xv = tf.reshape(Xv, shape=[-1, params.field_size, 1])
        embeddings_out = tf.multiply(Xi_embeddings, Xv, name='embeddings_out')  #[B, field_size, E]



    with tf.name_scope('interactive_attention'):
        element_wise_product_list = []
        for i in range(params.field_size):
            for j in range(i + 1, params.field_size):
                tmp_product = tf.multiply(embeddings_out[:, i, :], embeddings_out[:, j, :])
                element_wise_product_list.append(tmp_product)

        element_wise_product = tf.stack(element_wise_product_list)  #[num_comb, B, field_size, E]
        element_wise_product_trans = tf.transpose(element_wise_product, perm=[1, 0, 2]) #

        interaction = tf.reduce_sum(element_wise_product_trans, axis=2)
        num_interactions = int((params.field_size - 1) * params.field_size / 2)

        hidden_size0, hidden_size1 = 8, params.emb_dim

        attention_w = tf.get_variable(dtype=tf.float32,
                                      initializer=tf.initializers.glorot_normal,
                                      shape=[hidden_size1, hidden_size0],
                                      name='attention_w',
                                      )
        attention_mul = tf.matmul(tf.reshape(element_wise_product_trans, [-1, hidden_size1]), attention_w)
        attention_mul = tf.reshape(attention_mul, [-1, num_interactions, hidden_size0]) #[
        # attention_p = tf.get_variable(dtype=tf.float32,
        #                               initializer=tf.initializers.glorot_normal,
        #                               shape=[hidden_size0],
        #                               name='attention_p')
        # # attention_b = tf.get_variable(dtype=tf.float32,
        # #                               initializer=tf.initializers.glorot_normal,
        # #                               shape=[num_interactions, params.emb_dim],
        # #                               name='attention_b')
        # attention_relu = tf.multiply(attention_p, tf.nn.relu(attetion_mul))
        attention_mul = tf.nn.tanh(attention_mul)
        attention_relu = tf.layers.dense(attention_mul, 1)
        attention_softmax = tf.nn.softmax(attention_relu, axis=1, name='attention_matrix')
        # attention_softmax = tf.ones([params.batch_size, num_interactions, 1])
        # attention_softmax = attention_relu

        attention_out = tf.layers.dropout(inputs=attention_softmax, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
        afm = tf.reduce_sum(tf.multiply(attention_out, element_wise_product_trans), axis=1)
        afm_logit = tf.layers.dense(afm, 1)

        # with tf.name_scope('outputs'):
        #     output = y_first_order_logit + y_second_order_logit + afm_logit
    score = tf.nn.sigmoid(tf.identity(afm_logit), name='score')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(predictions=score, mode=mode)
    else:
        label1 = tf.identity(tf.reshape(labels, [-1, 1]), name='label1')
        label2 = tf.identity(tf.reshape(labels, [-1, 1]), name='label2')
        with tf.name_scope('metrics'):
            ctr_auc_score = tf.metrics.auc(labels=label1, predictions=score, name='ctr_auc_score')
            cvrctr_auc_score = tf.metrics.auc(labels=label2, predictions=score, name='cvrctr_auc_score')

        with tf.name_scope('loss'):
            ctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label1, predictions=score), name='ctr_loss')
            cvrctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label2, predictions=score), name='cvrctr_score')

            loss = tf.add(ctr_loss, cvrctr_loss, name='loss')
        metrics = {'ctr_auc_metric': ctr_auc_score, 'cvrctr_auc_metric': cvrctr_auc_score}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
        #optimizer = YFOptimizer(learning_rate=params.lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)


def model_estimator(params):
    tf.reset_default_graph()
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU}),
        log_step_count_steps=params.log_step_count_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_summary_steps=params.save_summary_steps,
        # train_distribute = tf.distribute.MirroredStrategy(),
        # eval_distribute = tf.distribute.MirroredStrategy(),
    )


    model = tf.estimator.Estimator(model_fn, config=config, model_dir=params.model_dir, params=params)

    return model
