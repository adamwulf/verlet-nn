//
//  StaticNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "StaticNeuron.h"
#import "AbstractNeuron+Protected.h"

@implementation StaticNeuron {
    CGFloat _value;
}

- (instancetype)initWithValue:(CGFloat)value
{
    if (self = [super init]) {
        _value = value;
    }
    return self;
}

- (CGFloat)activation
{
    return _value;
}

- (void)setActivation:(CGFloat)activation
{
    @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot set activation of StaticNeuron" userInfo:nil];
}

- (void)addInput:(AbstractNeuron *)neuron withWeight:(CGFloat)initialWeight
{
    @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot add an input to a static neuron" userInfo:nil];
}

@end
