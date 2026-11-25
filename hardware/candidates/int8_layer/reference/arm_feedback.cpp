// Check the PS side of every recorded RTL request with the factory ARM64 ORT.
#include "board/asr/ort_backend.cpp"
#include "../kernel/layer.hpp"
#include <cfenv>
#include <cstdio>
#include <map>

namespace {
using namespace edge_speech::asr;
struct FloatTensor {
    std::vector<int64_t> shape;
    std::vector<float> values;
};
using Signals = std::map<std::string, FloatTensor>;

struct Cut {
    Ort::Session session;
    std::vector<std::string> inputs;
    std::string output;
    Cut(Ort::Env& env, const std::string& file)
        : session(env, file.c_str(), make_options(false, true, 1)) {
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.emplace_back(session.GetInputNameAllocated(i, alloc).get());
        if (session.GetOutputCount() != 1) throw std::runtime_error("PS cut output count");
        output = session.GetOutputNameAllocated(0, alloc).get();
    }
    FloatTensor eval(Signals& signals, const Ort::MemoryInfo& cpu) {
        std::vector<Ort::Value> tensors;
        std::vector<const char*> names;
        for (const auto& name : inputs) {
            auto& x = signals.at(name);
            tensors.push_back(Ort::Value::CreateTensor<float>(cpu, x.values.data(), x.values.size(),
                                                               x.shape.data(), x.shape.size()));
            names.push_back(name.c_str());
        }
        const char* out = output.c_str();
        auto result = session.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(), tensors.size(), &out, 1);
        auto info = result[0].GetTensorTypeAndShapeInfo();
        const float* data = result[0].GetTensorData<float>();
        return {info.GetShape(), {data, data + info.GetElementCount()}};
    }
};
struct Layer {std::string id, input, output; int rows;};

void read_exact(std::istream& in, void* data, size_t size) {
    if (!in.read(static_cast<char*>(data), size)) throw std::runtime_error("short input to ARM replay");
}

int run(int argc, char** argv) {
    if (argc != 6 || std::fesetround(FE_TONEAREST)) throw std::runtime_error("ARM entry");
    std::ifstream contract("/layers.tsv");
    std::vector<Layer> layers;
    Layer spec;
    while (contract >> spec.id >> spec.rows >> spec.input >> spec.output) layers.push_back(spec);
    if (layers.size()!=10) throw std::runtime_error("bad layer list");
    auto env=make_env(argv[1]);
    auto cpu=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    Cut embed(env, argv[2]), finish(env, "/model/bridge-final.onnx");
    std::map<std::string,std::unique_ptr<Cut>> bridges;
    for (const auto& s : layers) {
        const auto file = "/model/bridge-"+s.id+".onnx";
        if (std::ifstream(file).good()) bridges[s.id] = std::make_unique<Cut>(env,file);
    }
    if (bridges.size()!=4) throw std::runtime_error("missing PS bridge");
    const std::string model=argv[3];
    Graph decoder(env, model + "/decoder-epoch-99-avg-1.onnx",
        {{{"y",integer,{1,2},0}},{{"decoder_out",floating,{1,projection},0}}},{},"decoder",false,true,1);
    Graph joiner(env, model + "/joiner-epoch-99-avg-1.onnx",
        {{{"encoder_out",floating,{1,projection},0},{"decoder_out",floating,{1,projection},0}},
         {{"logit",floating,{1,vocab_size},0}}},{},"joiner",false,true,1);
    std::array<int64_t,2> context{-1,0}, context_shape{1,2};
    auto y=Ort::Value::CreateTensor<int64_t>(cpu,context.data(),2,context_shape.data(),2);
    const OrtValue* token_input=y;
    auto decoded=decoder.run(&token_input);
    std::ifstream features(argv[4],std::ios::binary), edges("/edges.bin",std::ios::binary);
    int code_diffs=0, scale_diffs=0, checked=0;
    for(int c=0;c<32;++c) {
        Signals sig;
        sig["x"]={{1,39,80},std::vector<float>(39*80)};
        read_exact(features,sig["x"].values.data(),39*80*sizeof(float));
        sig[embed.output]=embed.eval(sig,cpu);
        for (const auto& s : layers) {
            if(bridges.count(s.id)) sig[s.input]=bridges.at(s.id)->eval(sig,cpu);
            auto& t=sig.at(s.input);
            if(t.shape!=std::vector<int64_t>{s.rows,1,256}) throw std::runtime_error("bad layer input shape");
            const size_t n=t.values.size();
            std::vector<int32_t> fixed(n);
            int64_t peak=0;
            for (size_t i=0;i<n;++i) {
                if(!std::isfinite(t.values[i])) throw std::runtime_error("nonfinite input");
                fixed[i]=int32_t(std::nearbyint(double(t.values[i])*65536.0));
                peak=std::max<int64_t>(peak,std::abs(int64_t(fixed[i])));
            }
            int32_t scale=int32_t(std::max<int64_t>(1,(peak+126)/127));
            int32_t recorded_scale, output_scale;
            std::vector<int8_t> saved(n), output(n);
            read_exact(edges,&recorded_scale,4); read_exact(edges,saved.data(),n);
            read_exact(edges,&output_scale,4); read_exact(edges,output.data(),n);
            int differences=0;
            for(size_t i=0;i<n;++i) differences+=int(fzint::divide(fixed[i],scale))!=saved[i];
            if(differences || scale!=recorded_scale)
                std::printf("EDGE_DIFF %d %s %d %d %d\n",c,s.id.c_str(),scale,recorded_scale,differences);
            code_diffs+=differences; scale_diffs+=scale!=recorded_scale; checked+=int(n);
            FloatTensor next{{s.rows,1,256},std::vector<float>(n)};
            for(size_t i=0;i<n;++i) next.values[i]=float(int64_t(output[i])*output_scale/65536.0);
            sig[s.output]=std::move(next);
        }
        auto encoded=finish.eval(sig,cpu);
        if(encoded.shape!=std::vector<int64_t>{1,8,projection}) throw std::runtime_error("bad final shape");
        const auto& port=joiner.spec.inputs.front();
        std::printf("ARM_FRAMES %d",c);
        for(int f=0;f<8;++f) {
            auto row=Ort::Value::CreateTensor<float>(cpu,encoded.values.data()+f*projection,projection,
                                                      port.shape.data(),port.shape.size());
            std::array<const OrtValue*,2> pair{row,decoded[0]};
            auto logits=joiner.run(pair.data());
            const float* first=logits[0].GetTensorData<float>();
            const auto token=std::max_element(first,first+vocab_size)-first;
            std::printf(" %ld",long(token));
            if(token!=0 && token!=2) {context={context[1],token}; decoded=decoder.run(&token_input);}
        }
        std::puts("");
    }
    if(edges.peek()!=EOF || features.peek()!=EOF) throw std::runtime_error("extra ARM input");
    std::printf("ARM_BINDINGS 320 %d %d %d\n",checked,code_diffs,scale_diffs);
    return 0;
}
}
int main(int argc,char** argv) {
    try {return run(argc,argv);}
    catch(const std::exception& e) {std::fprintf(stderr,"ARM_ERROR %s\n",e.what());return 1;}
}
