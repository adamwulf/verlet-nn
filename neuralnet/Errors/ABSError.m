//
//  ABSError.m
//  neuralnet
//
//  Created by Adam Wulf on 10/26/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "ABSError.h"
#import "AbstractNeuron.h"

@implementation ABSError

+ (instancetype)calculator
{
    return [[ABSError alloc] init];
}

- (CGFloat)errorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    return ABS([self simpleErrorFor:goal forNeuron:neuron]);
}

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    CGFloat derivABS;
    CGFloat derivSimpleError = -1;

    if ([self simpleErrorFor:goal forNeuron:neuron] > 0) {
        derivABS = 1;
    } else {
        derivABS = -1;
    }

    return derivSimpleError * derivABS;
}

@end
