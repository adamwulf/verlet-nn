//
//  WeightedSumNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "WeightedSumNeuron.h"
#import "AbstractNeuron+Protected.h"

@implementation WeightedSumNeuron

- (void)forwardPass
{
    CGFloat val = 0;
    for (int i = 0; i < [[self inputs] count]; i++) {
        AbstractNeuron *neuron = [self inputs][i];
        CGFloat weight = [[self weights][i] doubleValue];
        val += [neuron activation] * weight;
    }

    [self setActivation:val];
}

- (CGFloat)errorFor:(CGFloat)goal
{
    // when we do our forwardPass, our output is = input * weight.
    // our our error = f(weight) = (input * weight - goal)^2
    CGFloat err = [self activation] - goal;
    return err * err;
}

// we use our input, weight, output, goal to define our error.
// of these, we can only change our weight, so the calculation
// of error is an f(weight). This means to gradient decent to
// a better answer, we should take the derivative of the error function.
//
// raw error = input * weight - goal
//
// which we derive by: 1. [self simpleErrorFor:goal]
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
- (CGFloat)derivativeInputAtIndex:(NSInteger)neuronIndex andGoal:(CGFloat)goal
{
    AbstractNeuron *inputNeuron = [self inputs][neuronIndex];
    CGFloat weight = [[self weights][neuronIndex] doubleValue];
    CGFloat input = [inputNeuron activation];
    CGFloat derivative = 2 * input * (input * weight - goal);

#ifdef DEBUG
    CGFloat rawError = [self simpleErrorFor:goal];
    CGFloat dirAndAmount = rawError * [inputNeuron activation];

    NSAssert(derivative == dirAndAmount * 2, @"derivative is twice the dirAndAmount");
#endif

    return derivative;
}

@end
