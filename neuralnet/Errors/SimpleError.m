//
//  SimpleError.m
//  neuralnet
//
//  Created by Adam Wulf on 10/26/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "SimpleError.h"
#import "AbstractNeuron.h"

@implementation SimpleError

+ (instancetype)calculator
{
    return [[SimpleError alloc] init];
}

- (CGFloat)errorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    // when we do our forwardPass, our output is = input * weight.
    // our our error = f(weight) = (input * weight - goal)^2
    return goal - [neuron activation];
}

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    if ([self errorFor:goal forNeuron:neuron] > 0) {
        return -1;
    }

    return 1;
}

@end
