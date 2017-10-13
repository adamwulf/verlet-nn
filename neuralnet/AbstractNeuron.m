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

- (CGFloat)weightForNeuron:(AbstractNeuron *)neuron
{
    NSInteger index = [[self inputs] indexOfObject:neuron];

    if (index == NSNotFound) {
        @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot find weight for neuron that's not an input" userInfo:nil];
    }

    return [[self weights][index] doubleValue];
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
    [self backPropagateFor:goal withLearningRate:1.0];
}

// we use our input, weight, output, goal to define our error.
// of these, we can only change our weight, so the calculation
// of error is an f(weight). This means to gradient decent to
// a better answer, we should take the derivative of the error function.
//
// raw error = input * weight - goal
//
// which we derive by: 1. [self rawErrorFor:goal]
//                     2. [self output] - goal
//                     3. weight * input - goal
//
// and our mean squared error = f(weight) = (input * weight - goal) ^ 2
// so our error function is: 1. (iw - g)^2 = (iw - g)(iw - g)
//                           2. i^2w^2 - 2iwg - g^2
//                           3. 2i^2w - 2ig.
//
// the derivative of our error function should be used to calculate
// how much we should adjust our weights to correct for that error.
//
// the derivative of our error function is: 1. d/dw i^2w^2 - 2iwg - g^2
//                                          2. 2i^2w - 2ig
//                                          3. 2(wi - g)i
//                                          4. 2 * (weight * input - goal) * input
//
// Note: our dirAndAmount variable that we're using to correct our weight
// is equal to: 1. (raw error) * input
//              2. (weight * input - goal) * input
//
// and this is equal to exactly twice of our derivative function calcualted above (!)

- (void)backPropagateFor:(CGFloat)goal withLearningRate:(CGFloat)alpha
{
    NSAssert(alpha > 0, @"alpha > 0");
    NSAssert(alpha <= 1.0, @"alpha <= 1.0");

    for (int i = 0; i < [[self inputs] count]; i++) {
        AbstractNeuron *inputNeuron = [self inputs][i];
        CGFloat weight = [[self weights][i] doubleValue];
        CGFloat input = [inputNeuron output];
        CGFloat rawError = [self rawErrorFor:goal];
        CGFloat dirAndAmount = rawError * [inputNeuron output];
        CGFloat derivative = input * input * weight - 2 * input * goal;
        weight = weight - (dirAndAmount * alpha);
        [[self weights] replaceObjectAtIndex:i withObject:@(weight)];
    }
}

@end
