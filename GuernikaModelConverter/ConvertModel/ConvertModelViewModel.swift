//
//  ConvertModelViewModel.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import SwiftUI
import Combine

class ConvertModelViewModel: ObservableObject {
    @Published var showSuccess: Bool = false
    var showError: Bool {
        get { error != nil }
        set {
            if !newValue {
                error = nil
                isCoreMLError = false
            }
        }
    }
    var isCoreMLError: Bool = false
    @Published var error: String?
    
    var isReady: Bool {
        guard convertUnet || convertTextEncoder || convertVaeEncoder || convertVaeDecoder || convertSafetyChecker else {
            return false
        }
        switch modelOrigin {
        case .huggingface: return !huggingfaceIdentifier.isEmpty
        case .diffusers: return diffusersLocation != nil
        case .checkpoint: return checkpointLocation != nil
        }
    }
    @Published var process: ConverterProcess?
    var isRunning: Bool { process != nil }
    
    @Published var showOutputPicker: Bool = false
    @AppStorage("output_location") var outputLocation: URL?
    var modelOrigin: ModelOrigin {
        get { ModelOrigin(rawValue: modelOriginString) ?? .huggingface }
        set { modelOriginString = newValue.rawValue }
    }
    @AppStorage("model_origin") var modelOriginString: String = ModelOrigin.huggingface.rawValue
    @AppStorage("huggingface_identifier") var huggingfaceIdentifier: String = ""
    @AppStorage("diffusers_location") var diffusersLocation: URL?
    @AppStorage("checkpoint_location") var checkpointLocation: URL?
    var selectedModel: String? {
        switch modelOrigin {
        case .huggingface: return nil
        case .diffusers: return diffusersLocation?.lastPathComponent
        case .checkpoint: return checkpointLocation?.lastPathComponent
        }
    }
    
    var computeUnits: ComputeUnits {
        get { ComputeUnits(rawValue: computeUnitsString) ?? .cpuAndNeuralEngine }
        set { computeUnitsString = newValue.rawValue }
    }
    @AppStorage("compute_units") var computeUnitsString: String = ComputeUnits.cpuAndNeuralEngine.rawValue
    
    var compression: Compression {
        get { Compression(rawValue: compressionString) ?? .fullSize }
        set { compressionString = newValue.rawValue }
    }
    @AppStorage("compression") var compressionString: String = Compression.fullSize.rawValue
    
    @AppStorage("custom_size") var customSize: Bool = false
    @AppStorage("custom_size_width") var customWidth: Int = 512
    @AppStorage("custom_size_height") var customHeight: Int = 512
    
    @AppStorage("convert_unet") var convertUnet: Bool = true
    @AppStorage("chunk_unet") var chunkUnet: Bool = false
    @AppStorage("controlnet_support") var controlNetSupport: Bool = true
    @AppStorage("convert_text_encoder") var convertTextEncoder: Bool = true
    @AppStorage("load_embeddings") var loadEmbeddings: Bool = false
    @AppStorage("embeddings_location") var embeddingsLocation: URL?
    @AppStorage("convert_vae_encoder") var convertVaeEncoder: Bool = true
    @AppStorage("convert_vae_decoder") var convertVaeDecoder: Bool = true
    @AppStorage("convert_safety_checker") var convertSafetyChecker: Bool = false
    @AppStorage("precision_full") var precisionFull: Bool = false
    
    private var cancellables: Set<AnyCancellable> = []
    
    func start() {
        guard checkCoreMLCompiler() else {
            isCoreMLError = true
            error = "CoreMLCompiler not available.\nMake sure you have Xcode installed and you run \"sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/\" on a Terminal."
            return
        }
        do {
            let process = try ConverterProcess(
                outputLocation: outputLocation,
                modelOrigin: modelOrigin,
                huggingfaceIdentifier: huggingfaceIdentifier,
                diffusersLocation: diffusersLocation,
                checkpointLocation: checkpointLocation,
                computeUnits: computeUnits,
                customWidth: customSize && computeUnits == .cpuAndGPU ? customWidth : nil,
                customHeight: customSize && computeUnits == .cpuAndGPU ? customHeight : nil,
                convertUnet: convertUnet,
                chunkUnet: chunkUnet,
                controlNetSupport: controlNetSupport,
                convertTextEncoder: convertTextEncoder,
                embeddingsLocation: loadEmbeddings ? embeddingsLocation : nil,
                convertVaeEncoder: convertVaeEncoder,
                convertVaeDecoder: convertVaeDecoder,
                convertSafetyChecker: convertSafetyChecker,
                precisionFull: precisionFull,
                compression: compression
            )
            process.objectWillChange
                .receive(on: DispatchQueue.main)
                .sink { _ in
                    self.objectWillChange.send()
                }.store(in: &cancellables)
            NotificationCenter.default.publisher(for: Process.didTerminateNotification, object: process.process)
                .receive(on: DispatchQueue.main)
                .sink { _ in
                    if process.didComplete {
                        self.showSuccess = true
                    }
                    self.cancel()
                }.store(in: &cancellables)
            try process.start()
            withAnimation {
                self.process = process
            }
        } catch ConverterProcess.ArgumentError.noOutputLocation {
            showOutputPicker = true
        } catch ConverterProcess.ArgumentError.noHuggingfaceIdentifier {
            self.error = "Enter a valid identifier"
        } catch ConverterProcess.ArgumentError.noDiffusersLocation {
            self.error = "Enter a valid location"
        } catch ConverterProcess.ArgumentError.noCheckpointLocation {
            self.error = "Enter a valid location"
        } catch {
            print(error.localizedDescription)
            self.error = error.localizedDescription
            withAnimation {
                cancel()
            }
        }
    }
    
    func cancel() {
        withAnimation {
            process?.cancel()
            process = nil
        }
    }
    
    func selectAll() {
        convertUnet = true
        convertTextEncoder = true
        convertVaeEncoder = true
        convertVaeDecoder = true
        convertSafetyChecker = true
    }
    
    func selectNone() {
        convertUnet = false
        convertTextEncoder = false
        convertVaeEncoder = false
        convertVaeDecoder = false
        convertSafetyChecker = false
    }
    
    func checkCoreMLCompiler() -> Bool {
        let process = Process()
        process.executableURL = URL(string: "file:///usr/bin/xcrun")
        process.arguments = ["coremlcompiler", "version"]
        do {
            try process.run()
        } catch {
            Logger.shared.append("coremlcompiler not found")
            return false
        }
        return true
    }
}
