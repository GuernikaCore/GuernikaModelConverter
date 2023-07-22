//
//  ComputeUnits.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import Foundation

enum ComputeUnits: String, CaseIterable, Identifiable, CustomStringConvertible {
    case cpuAndNeuralEngine, cpuAndGPU, cpuOnly, all
    
    var id: String { rawValue }
    
    var description: String {
        switch self {
        case .cpuAndNeuralEngine: return "CPU and Neural Engine"
        case .cpuAndGPU: return "CPU and GPU"
        case .cpuOnly: return "CPU only"
        case .all: return "All"
        }
    }
    
    var shortDescription: String {
        switch self {
        case .cpuAndNeuralEngine: return "CPU & NE"
        case .cpuAndGPU: return "CPU & GPU"
        case .cpuOnly: return "CPU"
        case .all: return "All"
        }
    }
    
    var asCTComputeUnits: String {
        switch self {
        case .cpuAndNeuralEngine: return "CPU_AND_NE"
        case .cpuAndGPU: return "CPU_AND_GPU"
        case .cpuOnly: return "CPU_ONLY"
        case .all: return "ALL"
        }
    }
}
