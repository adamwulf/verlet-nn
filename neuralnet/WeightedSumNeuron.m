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
        val += [neuron output] * weight;
    }

    [self setCurrentValue:val];
}

@end
