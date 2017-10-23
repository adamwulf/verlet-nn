//
//  TraskMeanSquaredError.m
//  neuralnet
//
//  Created by Adam Wulf on 10/23/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "TraskMeanSquaredError.h"
#import "AbstractNeuron.h"

@implementation TraskMeanSquaredError

+ (instancetype)calculator
{
    return [[TraskMeanSquaredError alloc] init];
}

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron
{
    // compare to MeanSquaredError's implementation. This version
    // adds a 1/2 to the method to simplify the derivative, and
    // pushes the (-1) through.
    //
    // interestingly, the 1/2 isn't applied to the actual error calculation
    // (as that wouldn't look simple), but that's ok, because it's a simple
    // linear transform on the derivative, so the direction and relative
    // scale of the dervivative doesn't change. It's a "free" scale,
    // since this'll be multiplied by our learning rate anyway, it's just
    // like applying a slightly different learning rate.
    return [neuron activation] - goal;
}

@end
