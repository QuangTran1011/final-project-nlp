import tensorflow as tf
from tensorflow.keras import layers

class SupervisedContrastiveModel(tf.keras.Model):
    def __init__(self, backbone_model, temperature=0.05):
        super(SupervisedContrastiveModel, self).__init__()
        self.model = backbone_model
        self.temperature = temperature

    def call(self, inputs, training=False):
        # Generate embeddings
        embeddings = self.model(inputs, training=training)
        embeddings = embeddings[:, 0, :]  # Take the CLS token embedding
        if training:
            embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def compute_similarity_matrix(self, anchor, positive):
        # Compute cosine similarity
        anchor_normalized = tf.nn.l2_normalize(anchor, axis=1)
        positive_normalized = tf.nn.l2_normalize(positive, axis=1)
        similarity_matrix = tf.matmul(anchor_normalized, positive_normalized, transpose_b=True)
        return similarity_matrix

    def contrastive_loss(self, anchor, positive, negative):
        similarity_pos = self.compute_similarity_matrix(anchor, positive)
        similarity_neg = self.compute_similarity_matrix(anchor, negative)

        # Apply temperature scaling
        similarity_pos /= self.temperature
        similarity_neg /= self.temperature

        pos_numerator = tf.exp(similarity_pos)
        neg_numerator = tf.exp(similarity_neg)

        pos_row_sums = tf.reduce_sum(pos_numerator, axis=1)
        neg_row_sums = tf.reduce_sum(neg_numerator, axis=1)

        diagonal_elements = tf.linalg.diag_part(pos_numerator)

        result = diagonal_elements / tf.add(pos_row_sums, neg_row_sums)

        loss = -tf.reduce_mean(tf.math.log(result))

        # Compute contrastive loss
        return loss

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            anchor_embeddings = self(anchor)
            positive_embeddings = self(positive)
            negative_embeddings = self(negative)
            loss = self.contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}


class UnsupervisedSimCSE(tf.keras.Model):
    def __init__(self, model, temperature=0.05):
        super(UnsupervisedSimCSE, self).__init__()
        self.model = model
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.temperature = temperature

    
    def embedding_batch(self, inputs):
        input_ids = inputs["input_ids"]
        embeddings = self.model(input_ids)[:, 0, :]

        return embeddings


    def add_noise(self, embeddings):
        """
        Apply dropout (noise) to the embeddings to generate the anchor and positive samples.
        """

        # Apply dropout to generate noised embeddings (anchor and positive)
        noised_embeddings = self.dropout(embeddings, training=True)
        return noised_embeddings

    def compute_similarity_matrix(self, anchor, positive):
        """
        Compute cosine similarity matrix between anchor and positive embeddings.

        Args:
            anchor (tf.Tensor): Tensor of shape (N, D), anchor embeddings.
            positive (tf.Tensor): Tensor of shape (N, D), positive embeddings.

        Returns:
            tf.Tensor: Cosine similarity matrix of shape (N, N).
        """
        # Chuẩn hóa vector theo norm L2
        anchor_normalized = tf.nn.l2_normalize(anchor, axis=1)  # Shape: (N, D)
        positive_normalized = tf.nn.l2_normalize(positive, axis=1)  # Shape: (N, D)

        # Tính cosine similarity thông qua tích vô hướng
        similarity_matrix = tf.matmul(anchor_normalized, positive_normalized, transpose_b=True)  # Shape: (N, N)

        return similarity_matrix

    def contrastive_loss(self, anchor, positive):
        """
        Compute the contrastive loss between anchor and positive embeddings.
        """
        # Cosine similarity calculation
        similarity_matrix = self.compute_similarity_matrix(anchor, positive)  # Sử dụng self.compute_similarity_matrix

        # Apply temperature scaling
        similarity_matrix /= self.temperature

        # Calculate the numerator (similarity between anchor and positive of the same pair)
        numerator = tf.exp(similarity_matrix)

        row_sums = tf.reduce_sum(numerator, axis=1)

        diagonal_elements = tf.linalg.diag_part(numerator)

        result = diagonal_elements / row_sums

        loss = -tf.reduce_mean(tf.math.log(result))

        return loss

    @tf.function
    def train_step(self, data):
        inputs = data
        with tf.GradientTape() as tape:
            anchor = self.embedding_batch(inputs)
            positive = self.add_noise(anchor)
            loss = self.contrastive_loss(anchor, positive)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

class QqpTripletSupervisedContrastiveModel(tf.keras.Model):
    def __init__(self, backbone_model, temperature=0.05):
        super(SupervisedContrastiveModel, self).__init__()
        self.model = backbone_model
        self.temperature = temperature

    def call(self, inputs, training=False):
        # Generate embeddings
        if len(inputs.shape) == 3:
            inputs = tf.squeeze(inputs, axis=0)
        embeddings = self.model(inputs, training=training)
        embeddings = embeddings[:, 0, :]  # Take the CLS token embedding
        if training:
            embeddings = self.dropout(embeddings, training=training)
        return embeddings


    def compute_similarity_matrix(self, anchor, other_embeddings):
        # Compute cosine similarity
        anchor_normalized = tf.nn.l2_normalize(anchor, axis=1)
        other_normalized = tf.nn.l2_normalize(other_embeddings, axis=1)
        similarity_matrix = tf.matmul(anchor_normalized, other_normalized, transpose_b=True)
        return similarity_matrix

    def contrastive_loss(self, anchor, positive, negatives):
        # Compute similarity between anchor and positive
        similarity_pos = self.compute_similarity_matrix(anchor, positive)
        
        # Compute similarity between anchor and all negative samples
        similarity_neg = self.compute_similarity_matrix(anchor, negatives)
    
        # Apply temperature scaling
        similarity_pos /= self.temperature
        similarity_neg /= self.temperature
    
        # Compute exponentials of similarities
        pos_numerator = tf.exp(similarity_pos)
    
        # Compute sum over all negative samples (denominator part)
        neg_numerator = tf.exp(similarity_neg)
        neg_row_sums = tf.reduce_sum(neg_numerator, axis=1)
    
        # Compute the denominator, including the positive similarity part
        denominator = neg_row_sums + pos_numerator
    
        # Compute the contrastive loss using the formula
        result = pos_numerator / denominator
        
        # Final loss: negative log of the ratio
        loss = -tf.reduce_mean(tf.math.log(result))
    
        return loss

    
    def train_step(self, data):
        anchor, positive, negatives = data
        total_loss = 0.0  # Biến để lưu tổng loss
        
        with tf.GradientTape() as tape:
            # Lặp qua các mẫu riêng lẻ
            batch_size = tf.shape(anchor)[0]  # Lấy kích thước batch
            for i in range(batch_size):
                anchor_sample = anchor[i:i+1]  # Lấy một anchor sample
                positive_sample = positive[i:i+1]  # Lấy một positive sample
                negative_sample = negatives[i:i+1]  # Lấy một negative sample
                
                # Tính embeddings cho từng sample
                anchor_embeddings = self(anchor_sample)
                positive_embeddings = self(positive_sample)
                negative_embeddings = self(negative_sample)
        
                # Tính loss cho từng sample
                loss = self.contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                # Thêm loss vào tổng loss
                total_loss += loss
        
        # Tính gradients và cập nhật weights
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {"loss": total_loss}