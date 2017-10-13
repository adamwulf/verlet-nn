//
//  InputNeuron.h
//  neuralnet
//
//  Created by Adam Wulf on 10/12/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <neuralnet/neuralnet.h>

@interface InputNeuron : AbstractNeuron

@property(nonatomic, assign) CGFloat activation;

- (instancetype)initWithValue:(CGFloat)value;

@end
