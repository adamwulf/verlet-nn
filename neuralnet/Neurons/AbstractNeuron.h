//
//  AbstractNeuron.h
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import "ErrorCalculator.h"

@interface AbstractNeuron : NSObject

@property(nonatomic, readonly) CGFloat activation;

// the last recorded error for the neuron
@property(nonatomic, readonly) CGFloat delta;
@property(nonatomic, readonly) ErrorCalculator* errorCalculator;

-(instancetype) initWithErrorCalculator:(ErrorCalculator*)error;

- (void)addInput:(AbstractNeuron *)neuron withWeight:(CGFloat)initialWeight;

- (CGFloat)weightForNeuron:(AbstractNeuron *)neuron;

#pragma mark - Propagation

- (void)forwardPass;

- (void)backpropagateFor:(CGFloat)goal;

- (void)backpropagate;

- (void)updateWeightsWithAlpha:(CGFloat)alpha;

#pragma mark - Error

- (CGFloat)simpleErrorFor:(CGFloat)goal;

- (CGFloat)errorFor:(CGFloat)goal;

@end
