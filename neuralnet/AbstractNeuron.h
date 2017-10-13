//
//  AbstractNeuron.h
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface AbstractNeuron : NSObject

- (CGFloat)output;

- (void)addInput:(AbstractNeuron *)neuron withWeight:(CGFloat)initialWeight;

#pragma mark - Propagation

- (void)forwardPass;

- (void)backPropagateFor:(CGFloat)goal;

#pragma mark - Error

- (CGFloat)rawErrorFor:(CGFloat)goal;

- (CGFloat)meanSquaredErrorFor:(CGFloat)goal;

@end
