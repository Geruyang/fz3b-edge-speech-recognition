// Use the same ORT graph wrappers, contracts and CPU cuts as the frozen ARM replay.
#define main offline_replay_entry
#include "../reference/arm_feedback.cpp"
#undef main
#include "board/asr/fbank.hpp"
#include "layer_job.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <ctime>
#include <iostream>
#include <system_error>

namespace {
using namespace edge_speech::asr;
using fzint::Boundary;
using fzint::History;
static_assert(sizeof(Boundary)==LAYER_BOUNDARY && sizeof(History)==LAYER_HISTORY);
static_assert(sizeof(layer_job)==96);
uint64_t ns() {
    timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW,&ts)) throw std::runtime_error("clock");
    return uint64_t(ts.tv_sec)*1000000000+ts.tv_nsec;
}
struct Timed {
    uint64_t &sum, start;
    explicit Timed(uint64_t& total):sum(total),start(ns()) {}
    ~Timed(){sum+=ns()-start;}
};
struct Saved {Boundary input{},output{}; History next{}; layer_job job{}; uint64_t call{};};
constexpr uint64_t chunk_pcm = 32 * samples_per_frame;
constexpr uint64_t window_pcm = (39 - 1) * samples_per_frame + frame_end;

struct LiveAudio {
    std::deque<int16_t> pcm;
    uint64_t received=0;
    bool ended=false;

    void fill(uint64_t need) {
        std::array<int16_t,8192> buffer;
        while(!ended && received<need){
            uint32_t count=0;
            read_exact(std::cin,&count,sizeof(count));
            if(count==0){ended=true;break;}
            if(count>buffer.size()) throw std::runtime_error("PCM packet too large");
            read_exact(std::cin,buffer.data(),count*sizeof(int16_t));
            pcm.insert(pcm.end(),buffer.begin(),buffer.begin()+count);
            received+=count;
        }
    }
};
std::deque<int16_t> get_wave() {
    uint32_t bytes{};
    read_exact(std::cin,&bytes,4);
    if(bytes<44 || bytes>2000000) throw std::runtime_error("WAV length");
    std::vector<char> v(bytes);
    read_exact(std::cin,v.data(),bytes);
    auto u32=[&](size_t pos){uint32_t x; memcpy(&x,&v.at(pos),4);return x;};
    auto u16=[&](size_t pos){uint16_t x; memcpy(&x,&v.at(pos),2);return x;};
    if(memcmp(v.data(),"RIFF",4)||u32(4)!=bytes-8||memcmp(&v[8],"WAVE",4)) throw std::runtime_error("RIFF");
    bool fmt=false;
    for(size_t p=12;p+8<=v.size();){
        auto n=u32(p+4);
        if(n>v.size()-p-8) throw std::runtime_error("short RIFF chunk");
        if(!memcmp(&v[p],"fmt ",4))
            fmt=n>=16 && u16(p+8)==1 && u16(p+10)==1 && u32(p+12)==16000 && u16(p+22)==16;
        if(!memcmp(&v[p],"data",4)){
            if(!fmt || n!=160850*2) throw std::runtime_error("unexpected test audio format");
            std::vector<int16_t> pcm(n/2);
            memcpy(pcm.data(),v.data()+p+8,n);
            return {pcm.begin(),pcm.end()};
        }
        p+=8+n+(n&1);
    }
    throw std::runtime_error("missing PCM");
}
void pack(const FloatTensor &src,int rows,Boundary& boundary,int half) {
    if(src.shape!=std::vector<int64_t>{rows,1,256}) throw std::runtime_error("cut shape");
    std::array<int32_t,4096> fixed;
    int64_t peak=0;
    for(size_t i=0;i<src.values.size();++i){
        double value=std::nearbyint(double(src.values[i])*65536);
        if(!std::isfinite(value)||value<INT32_MIN||value>INT32_MAX) throw std::runtime_error("Q16 range");
        fixed[i]=int32_t(value);peak=std::max(peak,fzint::mag(fixed[i]));
    }
    int scale=boundary.scale[half]=fzint::scale_for(peak);
    for(int row=0;row<rows;++row) for(int c=0;c<256;++c)
        boundary.codes[half][row][c]=int8_t(fzint::divide(fixed[256*row+c],scale));
}
FloatTensor unpack(const Boundary& b,int half,int rows){
    FloatTensor t{{rows,1,256},std::vector<float>(rows*256)};
    for(int r=0;r<rows;++r) for(int k=0;k<256;++k)
        t.values[r*256+k]=float(int64_t(b.codes[half][r][k])*b.scale[half]/65536.0);
    return t;
}
template<class T> void raw(std::ostream& o,const T& v){
    o.write(reinterpret_cast<const char*>(&v),sizeof(v));
    if(!o) throw std::runtime_error("write capture");
}
std::string transcript(const std::vector<int>& tokens,const std::vector<std::string>& vocabulary){
    std::string text;
    for(auto t:tokens) text+=vocabulary.at(t);
    for(size_t p=0;(p=text.find(word_boundary,p))!=std::string::npos;++p)text.replace(p,3," ");
    if(auto p=text.find_first_not_of(' ');p!=std::string::npos)return text.substr(p,text.find_last_not_of(' ')-p+1);
    return {};
}

int live(uint64_t first_sequence,unsigned embedding_threads,bool stream=false) {
    const auto cold=ns();
    if(std::fesetround(FE_TONEAREST)) throw std::runtime_error("rounding");
    auto env=make_env("./libort.so");
    auto cpu=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    Graph embed(env,"model/embedding.onnx",
        {{{"x",floating,{1,39,80},0}},{{"/Transpose_output_0",floating,{16,1,256}}}},
        {},"embedding",false,true,embedding_threads);
    Cut finish(env,"model/bridge-final.onnx");
    std::ifstream contract("model/layers.tsv");
    std::vector<Layer> layers;
    std::map<std::string,std::unique_ptr<Cut>> bridges;
    Layer l;
    while(contract>>l.id>>l.rows>>l.input>>l.output){
        layers.push_back(l);
        auto path="model/bridge-"+l.id+".onnx";
        if(std::ifstream(path)) bridges[l.id]=std::make_unique<Cut>(env,path);
    }
    if(layers.size()!=10||bridges.size()!=4) throw std::runtime_error("contracts");
    Graph decoder(env,"model/decoder-epoch-99-avg-1.onnx",
        {{{"y",integer,{1,2},0}},{{"decoder_out",floating,{1,projection},0}}},{}, "decoder",false,true,1);
    Graph joiner(env,"model/joiner-epoch-99-avg-1.onnx",
        {{{"encoder_out",floating,{1,projection},0},{"decoder_out",floating,{1,projection},0}},
         {{"logit",floating,{1,vocab_size},0}}},{}, "joiner",false,true,1);
    auto vocabulary=load_tokens("model/tokens.txt");
    Fbank frontend(true);
    std::vector<Saved> saves(stream?0:160);
    std::vector<FeatureWindow> features(stream?0:32);
    std::vector<History> histories(layers.size());
    std::vector<int> frames,tokens;
    std::map<std::string,uint64_t> times;
    int fd=open("/dev/fz-layer-live",O_RDWR|O_CLOEXEC);
    if(fd<0) throw std::system_error(errno,std::generic_category(),"open layer");
    times["load"]=ns()-cold;
    std::cout<<"READY\n"<<std::flush;
    uint64_t start=ns();
    LiveAudio audio;
    if(!stream){
        audio.pcm=get_wave();
        audio.received=audio.pcm.size();
        audio.ended=true;
    }
    times["wait_receive_parse"]=ns()-start;
    const auto compute=ns();
    uint64_t offset=0;
    std::array<int64_t,2> context{-1,0},context_shape{1,2};
    auto y=Ort::Value::CreateTensor<int64_t>(cpu,context.data(),2,context_shape.data(),2);
    const OrtValue* token_in=y;
    auto decode=[&](){Timed t(times["decoder"]);return decoder.run(&token_in);};
    auto decoded=decode();
    std::string last_text;
    uint64_t expected_sequence=first_sequence;
    for(uint64_t chunk=0;;chunk+=2){
        audio.fill((chunk+1)*chunk_pcm+window_pcm);
        if(chunk*chunk_pcm>=audio.received) break;
        const int batch=(chunk+1)*chunk_pcm<audio.received?2:1;
        Signals sig[2];
        for(int h=0;h<batch;++h){
            uint64_t c=chunk+h;
            FeatureWindow feat;
            {
                Timed t(times["fbank"]);
                auto first=c*chunk_pcm;
                auto left=first?first-frame_left_context:0;
                while(offset<left){audio.pcm.pop_front();++offset;}
                feat=frontend.window(audio.pcm,offset,32*c,audio.received,first+window_pcm>audio.received);
                if(!stream) features.at(c)=feat;
            }
            Timed t(times["embedding"]);
            const auto& dims=embed.spec.inputs.front().shape;
            auto x=Ort::Value::CreateTensor<float>(cpu,feat.data(),feat.size(),dims.data(),dims.size());
            const OrtValue* input=x;
            auto result=embed.run(&input);
            const auto* data=result[0].GetTensorData<float>();
            sig[h][embed.spec.outputs.front().name]={{16,1,256},{data,data+4096}};
        }
        for(int i=0;i<10;++i){
            const auto& s=layers[i];
            Saved pending;
            auto& v=stream?pending:saves.at(chunk/2*10+i);
            for(int h=0;h<batch;++h){
                if(bridges.count(s.id)){
                    Timed t(times["bridge_"+s.id]); sig[h][s.input]=bridges.at(s.id)->eval(sig[h],cpu);
                }
                Timed t(times["pack"]);pack(sig[h].at(s.input),s.rows,v.input,h);
            }
            auto& j=v.job;
            j.layer=i;j.batch=batch;j.input=uintptr_t(&v.input);j.output=uintptr_t(&v.output);
            j.history=uintptr_t(&histories[i]);
            j.next_history=uintptr_t(&v.next);
            auto begin=ns();
            if(ioctl(fd,LAYER_RUN,&j)) throw std::system_error(errno,std::generic_category(),"layer "+s.id);
            v.call=ns()-begin;
            if((expected_sequence && j.sequence!=expected_sequence) || !j.sequence || j.result || (j.control&6)!=6)
                throw std::runtime_error("receipt order/status");
            expected_sequence=j.sequence+1;
            histories[i]=v.next;
            Timed t(times["unpack"]);
            for(int h=0;h<batch;++h) sig[h][s.output]=unpack(v.output,h,s.rows);
        }
        for(int h=0;h<batch;++h){
            auto& signals=sig[h];
            FloatTensor encoded;
            {Timed t(times["final_projection"]); encoded=finish.eval(signals,cpu);}
            if(encoded.shape!=std::vector<int64_t>{1,8,projection}) throw std::runtime_error("encoded shape");
            const auto& p=joiner.spec.inputs.front();
            for(int f=0;f<8;++f){
                int token;
                {
                    Timed t(times["joiner_argmax"]);
                    auto row=Ort::Value::CreateTensor<float>(cpu,&encoded.values[f*projection],projection,p.shape.data(),p.shape.size());
                    std::array<const OrtValue*,2> ins{row,decoded[0]};
                    auto out=joiner.run(ins.data());
                    auto logits=out[0].GetTensorData<float>();
                    token=std::max_element(logits,logits+vocab_size)-logits;
                }
                if(!stream) frames.push_back(token);
                if(token!=0 && token!=2){tokens.push_back(token);context={context[1],token};decoded=decode();}
            }
        }
        if(stream){
            auto text=transcript(tokens,vocabulary);
            if(text!=last_text){
                std::cout<<"PARTIAL\t"<<text<<"\n"<<std::flush;
                last_text=std::move(text);
            }
        }
    }
    auto text=transcript(tokens,vocabulary);
    times["board_compute"]=ns()-compute;
    close(fd);
    if(stream) std::cout<<"SAMPLES\t"<<audio.received<<"\n";
    std::cout<<"FINAL\t"<<text<<"\n"<<std::flush;
    if(stream) return 0;
    std::ofstream trace("trace.tsv"),io("io.bin",std::ios::binary),summary("times.tsv"),fbank("features.bin",std::ios::binary);
    trace<<"group\tlayer\tsequence\tioctl_ns\tcopy_in_ns\tsetup_ns\tstart_irq_ns\tirq_wake_ns\tcopy_out_ns\tcontrol\tresult\n";
    for(size_t i=0;i<saves.size();++i){
        const auto& v=saves[i];const auto& j=v.job;
        trace<<i<<'\t'<<layers[i%10].id<<'\t'<<j.sequence<<'\t'<<v.call<<'\t'<<j.copy_in_ns<<'\t'<<j.setup_ns<<'\t'<<j.start_irq_ns<<'\t'<<j.irq_wake_ns<<'\t'<<j.copy_out_ns<<'\t'<<j.control<<'\t'<<j.result<<'\n';
        raw(io,v.input);raw(io,v.output);raw(io,v.next);
    }
    for(const auto& f:features)raw(fbank,f);
    for(const auto& kv:times)summary<<kv.first<<'\t'<<kv.second<<'\n';
    for(int f:frames)std::cout<<f<<" ";
    std::cout<<"\n";
    return 0;
}
}
int main(int argc,char** argv){
    try{
        if(argc==1) return live(2,1);
        if(argc==2 && std::string(argv[1])=="--stream") return live(0,4,true);
        if(argc!=2 && argc!=3) throw std::invalid_argument("expected first physical job sequence and optional embedding thread count");
        std::string text=argv[1];
        if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("invalid job sequence");
        auto first=std::stoull(text);
        if(!first || first>UINT64_MAX-159) throw std::out_of_range("job sequence");
        const std::string threads=argc==3 ? argv[2] : "1";
        if(threads!="1" && threads!="4") throw std::invalid_argument("unsupported embedding thread count");
        return live(first,threads=="4"?4:1);
    }
    catch(const std::exception& e){std::cerr<<"LAYER_ERROR "<<e.what()<<"\n";return 1;}
}
