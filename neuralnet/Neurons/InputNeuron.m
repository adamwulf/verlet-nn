//
//  InputNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/12/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "InputNeuron.h"
#import "AbstractNeuron+Protected.h"

@implementation InputNeuron

@dynamic activation;

- (instancetype)initWithValue:(CGFloat)value
{
    if (self = [super init]) {
        [self setActivation:value];
    }
    return self;
}

- (CGFloat)transferFunction:(CGFloat)activation
{
    return activation;
}

- (CGFloat)transferDerivative
{
    return 1;
}

@end
