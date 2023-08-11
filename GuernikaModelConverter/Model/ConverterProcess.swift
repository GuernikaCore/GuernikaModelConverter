//
//  ConverterProcess.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import Combine
import Foundation

extension Process: Cancellable {
    public func cancel() {
        terminate()
    }
}

class ConverterProcess: ObservableObject {
    enum ArgumentError: Error {
        case noOutputLocation
        case noHuggingfaceIdentifier
        case noDiffusersLocation
        case noCheckpointLocation
    }
    enum Step: Int, Identifiable, CustomStringConvertible {
        case initialize = 0
        case loadEmbeddings
        case convertVaeEncoder
        case convertVaeDecoder
        case convertUnet
        case convertTextEncoder
        case convertSafetyChecker
        case convertControlNet
        case compressOutput
        case cleanUp
        case done
        
        var id: Int { rawValue }
        
        var description: String {
            switch self {
            case .initialize:
                return "Initialize"
            case .loadEmbeddings:
                return "Load embeddings"
            case .convertVaeEncoder:
                return "Convert Encoder"
            case .convertVaeDecoder:
                return "Convert Decoder"
            case .convertUnet:
                return "Convert Unet"
            case .convertTextEncoder:
                return "Convert Text encoder"
            case .convertSafetyChecker:
                return "Convert Safety checker"
            case .convertControlNet:
                return "Convert ControlNet"
            case .compressOutput:
                return "Compress"
            case .cleanUp:
                return "Clean up"
            case .done:
                return "Done"
            }
        }
    }
    struct StepProgress: CustomStringConvertible {
        let step: String
        let etaString: String
        var percentage: Float
        
        var description: String { "\(step).\(etaString)" }
    }
    
    let process: Process
    let steps: [Step]
    @Published var currentStep: Step = .initialize
    @Published var currentProgress: StepProgress?
    var didComplete: Bool {
        currentStep == .done && !process.isRunning
    }
    
    init(
        outputLocation: URL?,
        modelOrigin: ModelOrigin,
        huggingfaceIdentifier: String,
        diffusersLocation: URL?,
        checkpointLocation: URL?,
        computeUnits: ComputeUnits,
        customWidth: Int?,
        customHeight: Int?,
        convertUnet: Bool,
        chunkUnet: Bool,
        controlNetSupport: Bool,
        convertTextEncoder: Bool,
        embeddingsLocation: URL?,
        convertVaeEncoder: Bool,
        convertVaeDecoder: Bool,
        convertSafetyChecker: Bool,
        precisionFull: Bool,
        loRAsToMerge: [LoRAInfo] = [],
        compression: Compression
    ) throws {
        guard let outputLocation else {
            throw ArgumentError.noOutputLocation
        }
        var steps: [Step] = [.initialize]
        var arguments: [String] = [
            "-o", outputLocation.path(percentEncoded: false),
            "--bundle-resources-for-guernika",
            "--clean-up-mlpackages",
            "--compute-unit", "\(computeUnits.asCTComputeUnits)"
        ]
        
        switch modelOrigin {
        case .huggingface:
            guard !huggingfaceIdentifier.isEmpty else {
                throw ArgumentError.noHuggingfaceIdentifier
            }
            arguments.append("--model-version")
            arguments.append(huggingfaceIdentifier)
            arguments.append("--resources-dir-name")
            arguments.append(huggingfaceIdentifier)
        case .diffusers:
            guard let diffusersLocation else {
                throw ArgumentError.noDiffusersLocation
            }
            arguments.append("--model-location")
            arguments.append(diffusersLocation.path(percentEncoded: false))
            arguments.append("--resources-dir-name")
            arguments.append(diffusersLocation.lastPathComponent)
            arguments.append("--model-version")
            arguments.append(diffusersLocation.lastPathComponent)
        case .checkpoint:
            guard let checkpointLocation else {
                throw ArgumentError.noCheckpointLocation
            }
            arguments.append("--model-checkpoint-location")
            arguments.append(checkpointLocation.path(percentEncoded: false))
            arguments.append("--resources-dir-name")
            arguments.append(checkpointLocation.deletingPathExtension().lastPathComponent)
            arguments.append("--model-version")
            arguments.append(checkpointLocation.deletingPathExtension().lastPathComponent)
        }
        
        if computeUnits == .cpuAndGPU {
            arguments.append("--attention-implementation")
            arguments.append("ORIGINAL")
        } else {
            arguments.append("--attention-implementation")
            if #available(macOS 14.0, *) {
                arguments.append("SPLIT_EINSUM_V2")
            } else {
                arguments.append("SPLIT_EINSUM")
            }
        }
        
        if let embeddingsLocation {
            steps.append(.loadEmbeddings)
            arguments.append("--embeddings-location")
            arguments.append(embeddingsLocation.path(percentEncoded: false))
        }
        
        if convertVaeEncoder {
            steps.append(.convertVaeEncoder)
            arguments.append("--convert-vae-encoder")
        }
        if convertVaeDecoder {
            steps.append(.convertVaeDecoder)
            arguments.append("--convert-vae-decoder")
        }
        if convertUnet {
            steps.append(.convertUnet)
            arguments.append("--convert-unet")
            if chunkUnet {
                arguments.append("--chunk-unet")
            }
            if controlNetSupport {
                arguments.append("--controlnet-support")
            }
        }
        if convertTextEncoder {
            steps.append(.convertTextEncoder)
            arguments.append("--convert-text-encoder")
        }
        if convertSafetyChecker {
            steps.append(.convertSafetyChecker)
            arguments.append("--convert-safety-checker")
        }
        
        if !loRAsToMerge.isEmpty {
            arguments.append("--loras-to-merge")
            for loRA in loRAsToMerge {
                arguments.append(loRA.argument)
            }
        }
        
        if #available(macOS 14.0, *) {
            switch compression {
            case .quantizied6bit:
                arguments.append("--quantize-nbits")
                arguments.append("6")
                if convertUnet || convertTextEncoder {
                    steps.append(.compressOutput)
                }
            case .quantizied8bit:
                arguments.append("--quantize-nbits")
                arguments.append("8")
                if convertUnet || convertTextEncoder {
                    steps.append(.compressOutput)
                }
            case .fullSize:
                break
            }
        }
        
        if precisionFull {
            arguments.append("--precision-full")
        }
        
        if let customWidth {
            arguments.append("--output-w")
            arguments.append(String(describing: customWidth))
        }
        if let customHeight {
            arguments.append("--output-h")
            arguments.append(String(describing: customHeight))
        }
        
        let process = Process()
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = nil
        
        process.executableURL = Bundle.main.url(forAuxiliaryExecutable: "GuernikaTools")!
        print("Arguments", arguments)
        process.arguments = arguments
        self.process = process
        steps.append(.cleanUp)
        self.steps = steps
        
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.count > 0 {
                if let newLine = String(data: data, encoding: .utf8) {
                    self.handleOutput(newLine)
                }
            }
        }
    }
    
    init(
        outputLocation: URL?,
        controlNetOrigin: ModelOrigin,
        huggingfaceIdentifier: String,
        diffusersLocation: URL?,
        checkpointLocation: URL?,
        computeUnits: ComputeUnits,
        customWidth: Int?,
        customHeight: Int?
    ) throws {
        guard let outputLocation else {
            throw ArgumentError.noOutputLocation
        }
        let steps: [Step] = [.initialize, .convertControlNet, .cleanUp]
        var arguments: [String] = [
            "-o", outputLocation.path(percentEncoded: false),
            "--bundle-resources-for-guernika",
            "--clean-up-mlpackages",
            "--compute-unit", "\(computeUnits.asCTComputeUnits)"
        ]
        
        switch controlNetOrigin {
        case .huggingface:
            guard !huggingfaceIdentifier.isEmpty else {
                throw ArgumentError.noHuggingfaceIdentifier
            }
            arguments.append("--controlnet-version")
            arguments.append(huggingfaceIdentifier)
            arguments.append("--resources-dir-name")
            arguments.append(huggingfaceIdentifier)
        case .diffusers:
            guard let diffusersLocation else {
                throw ArgumentError.noDiffusersLocation
            }
            arguments.append("--controlnet-location")
            arguments.append(diffusersLocation.path(percentEncoded: false))
            arguments.append("--resources-dir-name")
            arguments.append(diffusersLocation.lastPathComponent)
            arguments.append("--controlnet-version")
            arguments.append(diffusersLocation.lastPathComponent)
        case .checkpoint:
            guard let checkpointLocation else {
                throw ArgumentError.noCheckpointLocation
            }
            arguments.append("--controlnet-checkpoint-location")
            arguments.append(checkpointLocation.path(percentEncoded: false))
            arguments.append("--resources-dir-name")
            arguments.append(checkpointLocation.deletingPathExtension().lastPathComponent)
            arguments.append("--controlnet-version")
            arguments.append(checkpointLocation.deletingPathExtension().lastPathComponent)
        }
        
        if computeUnits == .cpuAndGPU {
            arguments.append("--attention-implementation")
            arguments.append("ORIGINAL")
        } else {
            arguments.append("--attention-implementation")
            arguments.append("SPLIT_EINSUM_V2")
        }
        
        if let customWidth {
            arguments.append("--output-w")
            arguments.append(String(describing: customWidth))
        }
        if let customHeight {
            arguments.append("--output-h")
            arguments.append(String(describing: customHeight))
        }
        
        let process = Process()
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = nil
        
        process.executableURL = Bundle.main.url(forAuxiliaryExecutable: "GuernikaTools")!
        print("Arguments", arguments)
        process.arguments = arguments
        self.process = process
        self.steps = steps
        
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.count > 0 {
                if let newLine = String(data: data, encoding: .utf8) {
                    self.handleOutput(newLine)
                }
            }
        }
    }
    
    private func handleOutput(_ newLine: String) {
        let isProgressStep = newLine.starts(with: "\r")
        let newLine = newLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if isProgressStep {
            if
                let match = newLine.firstMatch(#": *?(\d*)%.*\|(.*)"#),
                let percentageRange = Range(match.range(at: 1), in: newLine),
                let percentage = Int(newLine[percentageRange]),
                let etaRange = Range(match.range(at: 2), in: newLine)
            {
                let etaString = String(newLine[etaRange])
                if newLine.starts(with: "Running MIL") || newLine.starts(with: "Running compression pass") {
                    currentProgress = StepProgress(
                        step: String(newLine.split(separator: ":")[0]),
                        etaString: etaString,
                        percentage: Float(percentage)
                    )
                } else {
                    currentProgress = nil
                }
            }
            DispatchQueue.main.async {
                Logger.shared.append(newLine, level: .success)
            }
            return
        }
        
        if
            let match = newLine.firstMatch(#"INFO:.*Converting ([^\s]+)$"#),
            let moduleRange = Range(match.range(at: 1), in: newLine)
        {
            let module = String(newLine[moduleRange])
            currentProgress = nil
            switch module {
            case "vae_encoder":
                currentStep = .convertVaeEncoder
            case "vae_decoder":
                currentStep = .convertVaeDecoder
            case "unet":
                currentStep = .convertUnet
            case "text_encoder":
                currentStep = .convertTextEncoder
            case "safety_checker":
                currentStep = .convertSafetyChecker
            case "controlnet":
                currentStep = .convertControlNet
            default:
                break
            }
        } else if let _ = newLine.firstMatch(#"INFO:.*Loading embeddings"#) {
            currentProgress = nil
            currentStep = .loadEmbeddings
        } else if let _ = newLine.firstMatch(#"INFO:.*Quantizing weights"#) {
            currentProgress = nil
            currentStep = .compressOutput
        } else if let _ = newLine.firstMatch(#"INFO:.*Bundling resources for Guernika$"#) {
            currentProgress = nil
            currentStep = .cleanUp
        } else if let _ = newLine.firstMatch(#"INFO:.*MLPackages removed$"#) {
            currentProgress = nil
            currentStep = .done
        } else if newLine.hasPrefix("usage: GuernikaTools") || newLine.hasPrefix("GuernikaTools: error: unrecognized arguments") {
            print(newLine)
            return
        } else {
            currentProgress = nil
        }
        DispatchQueue.main.async {
            Logger.shared.append(newLine)
        }
    }
    
    func start() throws {
        guard !process.isRunning else { return }
        Logger.shared.append("Starting python converter", level: .info)
        try process.run()
    }
    
    func cancel() {
        process.cancel()
    }
}

extension String {
    func firstMatch(_ pattern: String) -> NSTextCheckingResult? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        return regex.firstMatch(in: self, options: [], range: NSRange(location: 0, length: utf16.count))
    }
}
