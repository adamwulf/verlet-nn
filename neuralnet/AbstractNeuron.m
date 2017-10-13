//
//  AbstractNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "AbstractNeuron.h"
#import "AbstractNeuron+Protected.h"
#import "Constants.h"

@implementation AbstractNeuron

@synthesize inputs = _inputs;
@synthesize outputs = _outputs;
@synthesize weights = _weights;
@synthesize previousWeights = _previousWeights;
@synthesize activation = _activation;

- (instancetype)init
{
    if (self = [super init]) {
        _inputs = [NSMutableArray array];
        _outputs = [NSMutableArray array];
        _weights = [NSMutableArray array];
        _activation = 0;
    }
    return self;
}

- (void)addInput:(AbstractNeuron *)neuron withWeight:(CGFloat)initialWeight
{
    [_inputs addObject:neuron];
    [_weights addObject:@(initialWeight)];
    [neuron addOutput:self];
}

- (void)addOutput:(AbstractNeuron *)neuron
{
    [_outputs addObject:neuron];
}

- (CGFloat)weightForNeuron:(AbstractNeuron *)neuron
{
    NSInteger index = [[self inputs] indexOfObject:neuron];

    if (index == NSNotFound) {
        @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot find weight for neuron that's not an input" userInfo:nil];
    }

    return [[self weights][index] doubleValue];
}

- (CGFloat)previousWeightForNeuron:(AbstractNeuron *)neuron
{
    NSInteger index = [[self inputs] indexOfObject:neuron];

    if (index == NSNotFound) {
        @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot find weight for neuron that's not an input" userInfo:nil];
    }

    if (index >= [_previousWeights count]) {
        @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot find previous weight for neuron that's not an input" userInfo:nil];
    }

    return [[self previousWeights][index] doubleValue];
}

#pragma mark - Forward Propagation

- (void)forwardPass
{
    @throw kAbstractMethodException;
}

#pragma mark - Error

- (CGFloat)simpleErrorFor:(CGFloat)goal
{
    // if our value is lower than the goal,
    // then our error will be < 0, to signify
    // that we're below where we should be.
    return [self activation] - goal;
}

- (CGFloat)errorFor:(CGFloat)goal
{
    @throw kAbstractMethodException;
}

- (CGFloat)derivativeInputAtIndex:(NSInteger)neuronIndex andGoal:(CGFloat)goal
{
    @throw kAbstractMethodException;
}

#pragma mark - Gradient Descent

- (void)backpropWithLearningRate:(CGFloat)alpha
{
    CGFloat dErrTotaldOut = 0;
    for (int i = 0; i < [[self outputs] count]; i++) {
        AbstractNeuron *outNeuron = [self outputs][i];
        CGFloat deltaOutput = [outNeuron derivative];
        CGFloat outWeight = [outNeuron previousWeightForNeuron:self];
        CGFloat dErrOutdOut = (deltaOutput * outWeight);

        dErrTotaldOut += dErrOutdOut;
    }

    _derivative = dErrTotaldOut;

    [self backPropagateFor:dErrTotaldOut withLearningRate:alpha];
}

- (void)backPropagateFor:(CGFloat)goal
{
    _derivative = goal - [self activation];

    [self backPropagateFor:goal withLearningRate:1.0];
}

- (void)backPropagateFor:(CGFloat)goal withLearningRate:(CGFloat)alpha
{
    NSAssert(alpha > 0, @"alpha > 0");
    NSAssert(alpha <= 1.0, @"alpha <= 1.0");

    _previousWeights = [_weights copy];

    for (int i = 0; i < [[self inputs] count]; i++) {
        CGFloat weight = [[self weights][i] doubleValue];
        CGFloat derivative = [self derivativeInputAtIndex:i andGoal:goal];
        weight = weight - (derivative * alpha);
        [[self weights] replaceObjectAtIndex:i withObject:@(weight)];
    }
}

@end
