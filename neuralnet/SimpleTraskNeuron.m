//
//  SimpleTraskNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/10/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "SimpleTraskNeuron.h"
#import "AbstractNeuron+Protected.h"

// This neuron is based on the book Grokking Deep Learning by Andrew W. Trask.
// The derivative of the error function used in his book is half of the true
// derivative, so this class separates this change from the exact math.
@implementation SimpleTraskNeuron

- (CGFloat)derivativeNeuronAtIndex:(NSInteger)neuronIndex andGoal:(CGFloat)goal
{
    return [super derivativeNeuronAtIndex:neuronIndex andGoal:goal] / 2.0;
}

@end
