//
//  ErrorCalculator.m
//  neuralnet
//
//  Created by Adam Wulf on 10/23/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "ErrorCalculator.h"
#import "AbstractNeuron.h"
#import "Constants.h"

@implementation ErrorCalculator

+ (instancetype)calculator
{
    @throw kAbstractMethodException;
}

- (CGFloat)simpleErrorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    return goal - [neuron activation];
}

- (CGFloat)errorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    @throw kAbstractMethodException;
}

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    @throw kAbstractMethodException;
}

@end
