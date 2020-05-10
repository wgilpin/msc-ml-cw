from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import CSVLogger

from wame import WAME
from TimeHistory import TimeHistory

def create_model(x_shape, optimizer):
    inputs = Input(shape=x_shape)

    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    dense2 = Dense(128, activation='relu', )(dense1)
    dropped2 = Dropout(0.3)(dense2)
    dense3 = Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01))(dropped2)

    # prediction
    output_layer = Dense(6, activation='softmax')(dense3)

    #model
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print("3 layer model")
    return model

def eval_model(x_train, y_train, x_test, y_test, optimizer, epochs, optimizer_name):

    f_model = create_model(x_train.shape[1:], optimizer)

    import datetime
    today = datetime.datetime.now()
    csv_logger = CSVLogger(f'./logs/log-{optimizer_name}-{today:%Y-%m-%d %H-%M}.csv', append=True, separator=';')
    time_callback = TimeHistory()
    f_model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[time_callback, csv_logger])
    times = time_callback.get_times()

    score = f_model.evaluate(x_test, y_test, batch_size=40)
    return score, times

def tune_model(x_train, y_train, x_test, y_test):

    # define the grid search parameters
    batch_sizes = [40]
    epochs = [80, 100, 120]
    epsilons = [1e-4, 1e-5, 1e-6]

    best_params = {}
    best_score = 10000
    for batch_size in batch_sizes:
        for epoch in epochs:
            for epsilon in epsilons:
                f_model = create_model(x_train.shape[1:], optimizer=WAME(epsilon=epsilon))
                f_model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)
                score = f_model.evaluate(x_test, y_test, batch_size=40)
                if score<best_score:
                    best_score=score
                    best_params={"epochs": epoch, "batch_size":batch_size, "epsilon": epsilon}
    # summarize results
    print("Best: %f using %s" % (best_score, best_params))

