def create_Model():
    """
        Model architecture
    """
    input_shape = image_shape = (64,64,3)
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', name='conv_1',activation='LeakyReLU', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3,3), padding='same',name='conv_2',activation='LeakyReLU', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3,3), padding='same',name='conv_3',activation='LeakyReLU', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, name='dense1',activation='LeakyReLU'))
    model.add(layers.Dropout(.4))
    model.add(Dense(512, name='dense2',activation='LeakyReLU',kernel_regularizer=l2(1e-3)))
    #model.add(layers.Dropout(.2))
    
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(euclidean_distance)([encoded_l, encoded_r])
    #Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_layer)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
    
    siamese_net.compile(loss=loss(margin=1), optimizer="Adam",metrics=['acc'])
    #'binary_crossentropy'
    print(siamese_net.summary())
    
    # return the model
    return siamese_net


siamese1 = create_Model()
