//
//  ReluNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/20/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "ReluNeuron.h"

@implementation ReluNeuron

- (CGFloat)transferFunction:(CGFloat)activation
{
    if (activation > 0) {
        return activation;
    } else {
        return 0;
    }
}

// if I used an activation function beyond the weighted sum
// then i'd need to use the derivative function here. for example,
// the sigmoid activation function would mean:
// return [self activation] * (1.0 - [self activation]);
// but since instead of f(x) = sigmoid(x), i'm only using f(x) = x,
// so my derivative is 1.
- (CGFloat)transferDerivative
{
    if ([self activation] > 0) {
        return 1;
    } else {
        return 0;
    }
}

@end
