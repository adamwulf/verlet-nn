//
//  MeanSquaredError.m
//  neuralnet
//
//  Created by Adam Wulf on 10/23/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "MeanSquaredError.h"
#import "AbstractNeuron.h"

@implementation MeanSquaredError

+ (instancetype)calculator
{
    return [[MeanSquaredError alloc] init];
}

- (CGFloat)errorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    // when we do our forwardPass, our output is = input * weight.
    // our our error = f(weight) = (input * weight - goal)^2
    CGFloat err = [neuron activation] - goal;
    return err * err;
}

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    // (2 * (goal - [self activation]) * -1) is the derivative of (goal - [self activation])^2
    //
    // according to http://www.webpages.ttu.edu/dleverin/neural_network/neural_networks.html
    // "½ is a value applied to simplify the function’s derivative" which would mean
    // we could simplify the calculation below to:
    // ([self activation] - goal) * [self transferDerivative]
    // note: I've pushed the -1 through so that the goal and activation switch places
    // as well, but i've left it in below for clarity about the derivative.
    return (2 * (goal - [neuron activation]) * (-1));
}

@end
