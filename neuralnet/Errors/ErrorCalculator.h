//
//  ErrorCalculator.h
//  neuralnet
//
//  Created by Adam Wulf on 10/23/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@class AbstractNeuron;

@interface ErrorCalculator : NSObject

+ (instancetype)calculator;

- (CGFloat)errorFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron;

- (CGFloat)errorDerivativeFor:(CGFloat)goal forNeuron:(AbstractNeuron *)neuron;

@end
