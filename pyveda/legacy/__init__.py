def vedabase_batch_write(data, database=None, partition=[70, 20, 10]):
    trainp, testp, valp = partition
    images, labels, ids = data
    batch_size = images.shape[0]
    ntrain = round(batch_size * (trainp * 0.01))
    ntest = round(batch_size * (testp * 0.01))
    nval = round(batch_size * (valp * 0.01))

    # write training data, record ids
    database.train.images.append_batch(images[:ntrain])
    database.train.labels.append_batch(labels[:ntrain])
    database.train.id_table.append(ids[:ntrain])
    database.train.id_table.flush()

    # write testing data, record ids
    database.test.images.append_batch(images[ntrain:ntrain + ntest])
    database.test.labels.append_batch(labels[ntrain:ntrain + ntest])
    database.test.id_table.append(ids[ntrain:ntrain + ntest])
    database.test.id_table.flush()

    # write validation data, record ids
    database.validate.images.append_batch(images[ntrain + ntest:])
    database.validate.labels.append_batch(labels[ntrain + ntest:])
    database.validate.id_table.append(ids[ntrain + ntest:])
    database.validate.id_table.flush()


