//
//  ClampNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/20/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "ClampNeuron.h"
#import "AbstractNeuron+Protected.h"

@interface ClampNeuron ()

@property(nonatomic, assign) CGFloat min;
@property(nonatomic, assign) CGFloat max;
@property(nonatomic, strong) AbstractNeuron *neuron;

@end

@implementation ClampNeuron

- (instancetype)initWithNeuron:(AbstractNeuron *)neuron andMin:(CGFloat)min andMax:(CGFloat)max
{
    if (self = [super init]) {
        _min = min;
        _max = max;
        _neuron = neuron;
    }
    return self;
}

- (CGFloat)activation
{
    return MIN([self max], MAX([self min], [super activation]));
}

#pragma mark - AbstractNeuron

- (CGFloat)transferFunction:(CGFloat)activation
{
    return [[self neuron] transferFunction:activation];
}

// if I used an activation function beyond the weighted sum
// then i'd need to use the derivative function here. for example,
// the sigmoid activation function would mean:
// return [self activation] * (1.0 - [self activation]);
// but since instead of f(x) = sigmoid(x), i'm only using f(x) = x,
// so my derivative is 1.
- (CGFloat)transferDerivative
{
    return [[self neuron] transferDerivative];
}

@end
