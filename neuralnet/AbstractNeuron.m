//
//  AbstractNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "AbstractNeuron.h"
#import "AbstractNeuron+Protected.h"
#import "Constants.h"

@implementation AbstractNeuron

@synthesize inputs = _inputs;
@synthesize weights = _weights;
@synthesize currentValue = _currentValue;

- (instancetype)init
{
    if (self = [super init]) {
        _inputs = [NSMutableArray array];
        _weights = [NSMutableArray array];
        _currentValue = 0;
    }
    return self;
}

- (CGFloat)output
{
    return _currentValue;
}

- (void)addInput:(AbstractNeuron *)neuron withWeight:(CGFloat)initialWeight
{
    [_inputs addObject:neuron];
    [_weights addObject:@(initialWeight)];
}

- (void)forwardPass
{
    @throw kAbstractMethodException;
}

@end
