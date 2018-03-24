from kaffe.tensorflow import Network

class CaffeNet(Network):
    def setup(self):
        (self.feed('input')
             .conv(7, 7, 96, 2, 2, padding='VALID', name='conv1')
             .lrn(2, 0.00010000000474974513, 0.75, name='norm1')
             .max_pool(3, 3, 3, 3, name='pool1')
             .conv(5, 5, 256, 1, 1, name='conv2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 512, 1, 1, name='conv3')
             .conv(3, 3, 512, 1, 1, name='conv4')
             .conv(3, 3, 512, 1, 1, name='conv5')
             .max_pool(3, 3, 3, 3, name='pool5')
             .fc(4048, name='fc6')
             .fc(4048, name='fc7')
             .fc(7, relu=False, name='fc8_cat')
             .softmax(name='prob'))