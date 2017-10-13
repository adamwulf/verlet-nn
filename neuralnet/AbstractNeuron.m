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

#pragma mark - Forward Propagation

- (void)forwardPass
{
    @throw kAbstractMethodException;
}

#pragma mark - Error

- (CGFloat)rawErrorFor:(CGFloat)goal
{
    // if our value is lower than the goal,
    // then our error will be < 0, to signify
    // that we're below where we should be.
    return [self output] - goal;
}

- (CGFloat)meanSquaredErrorFor:(CGFloat)goal
{
    CGFloat err = [self rawErrorFor:goal];
    return err * err;
}

#pragma mark - Gradient Descent

- (void)backPropagateFor:(CGFloat)goal
{
    for (int i = 0; i < [[self inputs] count]; i++) {
        AbstractNeuron *neuron = [self inputs][i];
        CGFloat weight = [[self weights][i] doubleValue];

        CGFloat dirAndAmount = [self rawErrorFor:goal] * [neuron output];
        weight = weight - dirAndAmount;
        [[self weights] replaceObjectAtIndex:i withObject:@(weight)];
    }
}

@end
