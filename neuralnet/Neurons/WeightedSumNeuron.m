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

- (CGFloat)transferFunction:(CGFloat)activation
{
    return activation;
}

- (CGFloat)errorFor:(CGFloat)goal
{
    // when we do our forwardPass, our output is = input * weight.
    // our our error = f(weight) = (input * weight - goal)^2
    CGFloat err = [self activation] - goal;
    return err * err;
}

// if I used an activation function beyond the weighted sum
// then i'd need to use the derivative function here. for example,
// the sigmoid activation function would mean:
// return [self activation] * (1.0 - [self activation]);
// but since instead of f(x) = sigmoid(x), i'm only using f(x) = x,
// so my derivative is 1.
- (CGFloat)transferDerivative
{
    return 1;
}

@end
