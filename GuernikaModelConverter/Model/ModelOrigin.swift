//
//  ModelOrigin.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import Foundation

enum ModelOrigin: String, CaseIterable, Identifiable, CustomStringConvertible {
    case huggingface
    case diffusers
    case checkpoint
    
    var id: String { rawValue }
    
    var description: String {
        switch self {
        case .huggingface: return "🤗 Hugging Face"
        case .diffusers: return "📂 Diffusers"
        case .checkpoint: return "💽 Checkpoint"
        }
    }
}
