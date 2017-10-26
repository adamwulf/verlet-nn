//
//  AbstractNeuron.m
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "AbstractNeuron.h"
#import "AbstractNeuron+Protected.h"
#import "TraskMeanSquaredError.h"
#import "Constants.h"

@implementation AbstractNeuron

@synthesize inputs = _inputs;
@synthesize outputs = _outputs;
@synthesize weights = _weights;
@synthesize activation = _activation;

- (instancetype)initWithErrorCalculator:(ErrorCalculator *)error
{
    if (self = [super init]) {
        _inputs = [NSMutableArray array];
        _outputs = [NSMutableArray array];
        _weights = [NSMutableArray array];
        _activation = 0;
        _errorCalculator = error;
    }
    return self;
}

- (instancetype)init
{
    if (self = [super init]) {
        _inputs = [NSMutableArray array];
        _outputs = [NSMutableArray array];
        _weights = [NSMutableArray array];
        _activation = 0;
        _errorCalculator = [[TraskMeanSquaredError alloc] init];
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


#pragma mark - Forward Propagation

- (void)forwardPass
{
    CGFloat val = 0;
    for (int i = 0; i < [[self inputs] count]; i++) {
        AbstractNeuron *neuron = [self inputs][i];
        CGFloat weight = [[self weights][i] doubleValue];
        val += [neuron activation] * weight;
    }

    _activation = [self transferFunction:val];
}

- (CGFloat)transferFunction:(CGFloat)activation
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
    return [[self errorCalculator] errorFor:goal forNeuron:self];
}

#pragma mark - Gradient Descent

- (void)backpropagate
{
    if (![[self outputs] count]) {
        @throw [NSException exceptionWithName:@"NeuronException" reason:@"Cannot backpropagate a neuron that does not have outputs" userInfo:nil];
    }

    CGFloat dErrTotaldOut = 0;
    for (int i = 0; i < [[self outputs] count]; i++) {
        AbstractNeuron *outNeuron = [self outputs][i];
        CGFloat deltaOutput = [outNeuron delta];
        CGFloat outWeight = [outNeuron weightForNeuron:self];
        CGFloat dErrOutdOut = (deltaOutput * outWeight);

        dErrTotaldOut += dErrOutdOut;
    }

    _delta = dErrTotaldOut * [self transferDerivative];
}

- (void)backpropagateFor:(CGFloat)goal
{
    _delta = [[self errorCalculator] errorDerivativeFor:goal forNeuron:self] * [self transferDerivative];
}

- (void)updateWeightsWithAlpha:(CGFloat)alpha
{
    NSAssert(alpha > 0, @"alpha > 0");
    NSAssert(alpha <= 1.0, @"alpha <= 1.0");

    CGFloat total = 0;

    for (int i = 0; i < [[self inputs] count]; i++) {
        AbstractNeuron *input = [self inputs][i];
        CGFloat weight = [[self weights][i] doubleValue];
        weight -= alpha * [self delta] * [input activation];
        [self updateWeightAtIndex:i with:weight];

        total += ABS(weight);
    }
}

- (void)updateWeightAtIndex:(NSInteger)index with:(CGFloat)weight
{
    [[self weights] replaceObjectAtIndex:index withObject:[NSNumber numberWithDouble:weight]];
}

- (CGFloat)transferDerivative
{
    @throw kAbstractMethodException;
}

@end
