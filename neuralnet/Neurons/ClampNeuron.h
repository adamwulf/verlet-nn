//
//  ClampNeuron.h
//  neuralnet
//
//  Created by Adam Wulf on 10/20/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <neuralnet/neuralnet.h>

@interface ClampNeuron : AbstractNeuron

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithNeuron:(AbstractNeuron *)neuron andMin:(CGFloat)min andMax:(CGFloat)max;

@end
