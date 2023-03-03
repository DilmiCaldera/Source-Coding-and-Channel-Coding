%% 
% *A. Developing a Function for Huffman Coding Algorithm*
% 
% 1.

%makecode function: appendix-->Functions-A(1)
sig = [1 2 3]; 
p = [ 0.4 0.5 0.1];
makecode(p)
%% 
% 2. For any number of symbols
% 
% Function is tested for 4, 5, 6 symbols with selected probabilities as follows.
%%
%makecode function: appendix-->Functions-A(2)
%3 symbols in part(1)
%4 symbols
sig4 = [1 2 3 4]; 
p4 = [ 0.25 0.5 0.1 0.15];
makecode(p4)
%5 symbols
sig5 = [1 2 3 4 5]; 
p5 = [ 0.25 0.25 0.2 0.15 0.15];
makecode(p5)
%6 symbols
sig6 = [1 2 3 4 5 6]; 
p6 = [ 0.3 0.25 0.15 0.12 0.1 0.08];
makecode(p6)
%% 
% 3. Huffman code for the English alphabet
%%
f_alphabet = [8.16, 1.492, 2.782, 4.153, 13.004, 2.228, 2.015, 6.094, 6.966, 0.153, 0.778, 4.025, 2.406,...
    6.749, 7.507, 1.929, 0.095, 5.787, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074];   %frequencies
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];

tot = sum(f_alphabet);
p_alphabet = [];    %probability of alphabet
for i = 1 : length(f_alphabet)
    p_alphabet(i) = f_alphabet(i)/tot;
end
hf_codes = makecode(p_alphabet);    %get huffman code for the alphabet

for i = 1 : length(alphabet)
    fprintf('%c: %s\n',alphabet(i),hf_codes(i))
end
%% 
% Discuss on the  codewords assigned for the different letters.
% 
% * Huffman code is an optimal prefix code, been free of prefixes. In the assigned 
% codewords, we can observe that the least frequent letters like 'q','z' have 
% longest codewords while the highest frequent letters like 'e','t' have smallest 
% length of codewords. Therefore the basic aim of coding is satisfied.
% 
% Comment on the efficiency of the code.
%%
Hx = 0; %entropy
L = 0; %average codeword length
for i = 1 : length(p_alphabet)
    Hx = Hx + (-p_alphabet(i)*log2(p_alphabet(i)));
    L = L + length(convertStringsToChars(hf_codes(i))) * p_alphabet(i); 
end
code_efficiency = Hx/L
%% 
% Therefore the code efficiency $\approx$1. So the code is efficient and 
% therefore optimum.
% 
% 4. Last name encoding
%%
text = 'caldera';   %last name encoding
%huffman_encoder function: appendix-->Functions-A(4)
code_enc = huffman_encoder(text,alphabet,hf_codes)
%% 
% *B. Arithmetic Coding Algorithms *
% 
% 1. Basic arithmetic encoding algorithm
%%
source = ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9'];             %source
p = [0.1, 0.1, 0.05, 0.05, 0.2, 0.1, 0.05, 0.05, 0.05, 0.25];   %probabilities
s = '180079';                                                   %input sequence

CDF = cdf(p);                                                   %cdf to given probabilities
X = x(s);                                                       %proper sequence

%appendix-->Functions-B(1)
%% 
% 2. Tag value for the index number

%appendix-->Functions-B(2)
tag = basic_arithmetic_code(CDF, X)
%% 
% 3. Binary codeword for the tag

%appendix-->Functions-B(3)
binary_codeword_tag = convert_to_binary(tag, p, X)
%% 
% 4. Incremental encoding scheme with interval scaling

%appendix-->Functions-B(1)
binary_tag_incr = incremental_arithmetic_code(CDF, X)
%% 
% 5.  Propose an incremental decoding algorithm for the encoding algorithm 
% in part 4, and  represent it as a pseudocode
% 
% $$\begin{array}{l}X=\left\lbrack X_1 ,X_2 ,\ldotp \ldotp \ldotp ,X_n \right\rbrack 
% \\\mathit{input}\longleftarrow T\left(X\right),\;\;\;F\left(X_1 \right),F\left(X_2 
% \right),\ldotp \ldotp \ldotp ,F\left(X_n \right)\;\;\;\;\;\;\;\;\;\;\mathit{binary}\;\mathit{tag},\mathit{CDF}\;\mathit{of}\;\mathit{Xs}\\\mathit{initialize}\longleftarrow 
% l^{\;0} =0,{\;u}^{\;0} =1\\\mathit{for}\;\mathit{each}\;k\;\mathit{do}\\\;\;\;\;\;\;\;\;\;\;\overset{~}{T} 
% \left(X\right)=\frac{T\left(X\right)-l^{\;k-1\;} }{u^{\;k-1} -l^{\;k-1} }\\\;\;\;\;\;\;\;\;\;\;\mathit{find}\;X_k 
% \;\;\mathit{such}\;\mathit{that}\;\;\;\;F\left(X_k -1\right)\le \overset{~}{T} 
% \left(X\right)\;<\;F\left(X_k \right)\\\;\;\;\;\;\;\;\;\;\;\mathit{update}\;l^{\;k} 
% ,u^{\;k} \\\;\;\;\;\;\;\;\;\;\;l^{\;k} \longleftarrow F\left(X_k -1\right)\\\;\;\;\;\;\;\;\;\;\;u^{\;k} 
% \longleftarrow F\left(X_k \right)\\\;\;\;\;\;\;\;\;\;\mathit{repeat}\;\mathit{until}\;\mathit{end}\;\mathit{of}\;\mathit{sequence}\;\;\;\;\;\;\;\;\;\end{array}$$
% 
% *C. Dictionary Coding Using Lempel-Ziv-Welh Algorithm*
% 
% 1. A pseudocode for the LZW encoding algorithm
% 
% $$\begin{array}{l}\mathit{initialize}\;\mathit{the}\;\mathit{static}\;\mathit{dictionary}\;\mathit{with}\;\mathit{single}\;\mathit{characters}\\\mathit{CC}=\mathit{current}\;\mathit{character}\\\mathit{while}\;\mathit{not}\;\mathit{end}\;\mathit{of}\;\mathit{input}\;\mathit{sequence}\\\;\;\;\;\mathit{NC}=\mathit{next}\;\mathit{character}\\\;\;\;\;\mathit{if}\;\mathit{CC}+\mathit{NC}\;\mathit{is}\;\mathit{in}\;\mathit{static}\;\mathit{dictionary}\;\\\;\;\;\;\;\;\;\;\;\;\;\mathit{CC}=\mathit{CC}+\mathit{NC}\\\;\;\;\;\mathit{else}\\\;\;\;\;\;\;\;\;\;\;\;\mathit{output}\;\mathit{the}\;\mathit{code}\;\mathit{for}\;\mathit{CC}\\\;\;\;\;\mathit{add}\;\mathit{CC}+\mathit{NC}\;\mathit{to}\;\mathit{the}\;\mathit{static}\;\mathit{dictionary}\\\;\;\;\;\;\;\;\;\;\;\;\mathit{CC}=\mathit{NC}\;\\\;\;\;\;\mathit{end}\;\mathit{while}\\\mathit{output}\;\mathit{code}\;\mathit{for}\;\mathit{CC}\end{array}$$
% 
% 
% 
% 2. LZW encode
%%
% static dictionary
index = [1 2 3 4 5 6];
strings = {'a','b','h','i','s','t'};
stat_dic = containers.Map(strings,index);
msg = 'thisahatbsahahah';
%appendix-->Functions-C(2)
[enc_out, dict_upd] = lzw_encoder(stat_dic, msg);      %LZW encoding

disp('output index sequence for the given string'); 
disp(enc_out); 
value = cell2mat(values(dict_upd));
key = keys(dict_upd);
[~,i] = sort(value);
val=sort(value); %dictionary values
key = key(i);   %dictionary keys
%cell2table(key)
for i = 1 : length(val)
    fprintf('%d: %s\n',val(i),key{i})
end
%% 
% 3. A decoding algorithm and the pseudocode
% 
% $$\begin{array}{l}\mathit{initialize}\;\mathit{static}\;\mathit{dictionary}\;\mathit{with}\;\mathit{single}\;\mathit{character}\;\mathit{strings}\\\mathit{PREV}=\mathit{code}\;\mathit{of}\;\mathit{first}\;\mathit{input}\\\mathit{output}\;\mathit{translation}\;\mathit{of}\;\mathit{PREV}\\\mathit{while}\;\mathit{not}\;\mathit{end}\;\mathit{of}\;\mathit{input}\;\mathit{stream}\\\;\;\;\;\;\mathit{NEW}=\mathit{next}\;\mathit{input}\;\mathit{code}\\\;\;\;\;\;\mathit{if}\;\mathit{NEW}\;\mathit{is}\;\mathit{not}\;\mathit{in}\;\mathit{dictionary}\\\;\;\;\;\;\;\;\;\;\;\mathit{TP}=\mathit{translation}\;\mathit{of}\;\mathit{PREV}\\\;\;\;\;\;\;\;\;\;\;\mathit{TP}=\mathit{TP}+\mathit{FC}\\\;\;\;\;\;\mathit{else}\\\;\;\;\;\;\;\;\;\;\;\mathit{TP}=\mathit{translation}\;\mathit{of}\;\mathit{NEW}\\\;\;\;\;\;\mathit{output}\;\mathit{TP}\\\;\;\;\;\;\mathit{FC}=\mathit{first}\;\mathit{character}\;\mathit{of}\;\mathit{TP}\\\;\;\;\;\;\mathit{PREV}+\mathit{FC}\;\mathit{to}\;\mathit{the}\;\mathit{dictionary}\\\;\;\;\;\;\mathit{PREV}=\mathit{NEW}\\\mathit{endwhile}\end{array}$$
% 
% 
% 
% 4.Decode the encoded index sequence

dic = containers.Map(index,strings);
%appendix-->Functions-C(4)
[msg_out, dict_final] = lzw_decoder(dic, enc_out);
msg_out     %decoded output
%% 
% *D. Transmission of data over an AWGN channel*
% 
% 1. Baseband communication system
%%
% Transmit a 10-symbol frame
txData_un = randi([0 1],10,1);             % Generate data
modSig_un = pskmod(txData_un,2,pi/2);      % Modulate
snr_rnd = 5;                               % a random value
rxSig_un = awgn(modSig_un,snr_rnd);        % Pass through AWGN
rxData_un = pskdemod(rxSig_un,2,pi/2);     % Demodulate
%appendix-->Functions-D
err_un = error(txData_un, rxData_un);
BER_un = err_un/length(txData_un)
%% 
% 2. Transmit powers corresponding to  different $\frac{\textrm{E}b}{\textrm{N0}}$ 
% values
%%
var = 0.1;          %AWGN variance
N0 = 2*var;         %var=N0/2
Eb_N0 = -4:2:12;    %given values
power = [];
for i = 1 : length(Eb_N0)
    Eb = N0*10^(Eb_N0(i)/10);
    power(i) = Eb;  %bit rate = 1
end
A = [Eb_N0; power]; %two arrays for two rows of the table
T = array2table(A,'RowNames',{'Eb/N0 (dB)','PT (W)'})   %table
%% 
% 3. 
%%
Eb_N0 = -4:2:12;    %given values

BER_un=zeros(1,length(Eb_N0));
bit_len = 1;
for i=1:length(Eb_N0)
    bit_count_un = 0;
    err_count_un = 0;
    while err_count_un<100
        txData_un = randi([0 1],1,bit_len);        % Generate data
        modSig_un = pskmod(txData_un,2,pi/2);      % Modulate
        snr = Eb_N0(i);                              
        rxSig_un = awgn(modSig_un,snr);            % Pass through AWGN
        rxData_un = pskdemod(rxSig_un,2,pi/2);     % Demodulate
        %appendix-->Functions-D
        err_count_un = err_count_un + error(txData_un, rxData_un);
        bit_count_un = bit_count_un + length(txData_un);
        if bit_count_un>=1000000
            break
        end
    end
    BER_un(i)= err_count_un/bit_count_un;
end
BER_un
%% 
% *E. Error correction using Hamming codes *
% 
% 1.  Baseband communication system
%%
%(7,4) Hamming code
n = 7;  %no.of total bits
k = 4;  %no.of message bits

p = [1 1 0;0 1 1;1 1 1;1 0 1];
G = [eye(k) p];
H = [p.' eye(n-k)];
t = syndtable(H);

%(1)
%for a random snr value
snr_rnd = 5;
%appendix-->Functions-E
err_hm7_4 = process(k, G, H, t, snr_rnd);
BER_hm7_4 = err_hm7_4/n
%% 
% 2. Simulation for each $\frac{\textrm{E}b}{\textrm{N0}}$ value
%%
%(2)
Eb_N0 = -4:2:12;    %given values

BER_hm7_4=zeros(1,length(Eb_N0));
bit_len = 1;
for i=1:length(Eb_N0)
    bit_count = 0;
    err_count = 0;
    while err_count<100
        snr = Eb_N0(i); 
        %appendix-->Functions-E
        error_out = process(k, G, H, t, snr);
        err_count = err_count + error_out;
        bit_count = bit_count + n;  
        if bit_count>=1000000
            break
        end
    end
    BER_hm7_4(i)= err_count/bit_count;
end

%(3)
BER_hm7_4
%% 
% 3. Average BER for each $\frac{\textrm{E}b}{\textrm{N0}}$ value
%%
BER_hm7_4
%% 
% 4. For  (15,11) Hamming code
%%
%(15,11) Hamming code
n = 15;  %no.of total bits
k = 11;  %no.of message bits

p = [1 1 0 0;0 1 1 0;0 0 1 1;1 0 1 0;1 0 0 1;0 1 0 1;1 1 1 0;0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 1];
G = [eye(k) p];
H = [p.' eye(n-k)];
t = syndtable(H);

%(1)
%for a random snr value
snr_rnd = 5;
%appendix-->Functions-E
err_hm15_11 = process(k, G, H, t, snr_rnd);
BER_hm15_11 = err_hm15_11/n
%(2)
Eb_N0 = -4:2:12;    %given values

BER_hm15_11=zeros(1,length(Eb_N0));
bit_len = 1;
for i=1:length(Eb_N0)
    bit_count = 0;
    err_count = 0;
    while err_count<100
        snr = Eb_N0(i);
        %appendix-->Functions-E
        error_out = process(k, G, H, t, snr);
        err_count = err_count + error_out;
        bit_count = bit_count + n;  
        if bit_count>=1000000
            break
        end
    end
    BER_hm15_11(i)= err_count/bit_count;
end

%(3)
BER_hm15_11
%% 
% 5.  Average BER values obtained for the (7,4), (15,11) Hamming coded systems 
% and  for the uncoded system
%%
Eb_N0 = -4:2:12;    %given values
%avg BER values
BER_un = [0.1637    0.1212    0.0895    0.0404    0.0143    0.0022    0.0002    0.0000         0];
BER_hm7_4 = [0.2342    0.1104    0.0486    0.0118    0.0012    0.0001         0         0         0];
BER_hm15_11 = [0.2368    0.1333    0.0709    0.0176    0.0033    0.0001    0.0000         0         0];

semilogy(Eb_N0,BER_un,'b')
hold on
semilogy(Eb_N0,BER_hm7_4,'r')
hold on
semilogy(Eb_N0,BER_hm15_11,'g')
grid
legend('Uncoded','Hamming coded (7,4)','Hamming coded (15,11)')
xlabel('Eb/N0 (dB)')
ylabel('Bit Error Rate')
hold off
%% 
% 6. Theoretical expressions for the BER of the uncoded and coded systems
% 
% * *Uncoded system*
% 
% Received signal (y) in a BPSK communication system,
% 
% $y=x+n\;\;$   considering the real part of n,   $y=x+n_{R\;\;\;\;}$
% 
% where $x$: transmitted signal and belongs to $\left\lbrace \sqrt{E_b \;},\;-\sqrt{E_b 
% }\right\rbrace$
% 
% $n_R$: real AWGN with a given variance and $n_R ~N\left(0,\frac{N_0 }{2}\right)$
% 
% Bit Error Rate of uncoded systems $\left(\mathrm{BER}=P_{\mathrm{bu}} =P_{\mathrm{eu}} 
% \right)$,
% 
% $$\begin{array}{l}\mathrm{BER}=P\left(y<0,x=\sqrt{E_b }\right)+P\left(y>0,x=-\sqrt{E_b 
% }\right)\\\;\;\;\;\;\;\;\;\;\;=2P\left(y<0,x=\sqrt{E_b }\right)\\\;\;\;\;\;\;\;\;\;\;=2P\left(x+n_{R\;\;\;\;} 
% <0,x=\sqrt{E_b }\right)\\\;\;\;\;\;\;\;\;\;\;=2P\left(n<-\sqrt{E_b }\right)\ldotp 
% P\left(x=-\sqrt{E_b }\right)\\\;\;\;\;\;\;\;\;\;\;=2Q\left(\sqrt{\frac{2E_b 
% }{N_0 }}\right)\left(\frac{1}{2}\right)\\\;\;\;\;\;\;\;\;\;\;=Q\left(\sqrt{\frac{2E_{\mathrm{bu}} 
% }{N_0 }}\right)\end{array}$$
% 
% 
% 
% * *Coded system*
% 
% Raw error rate of coded systems,    $P_{\mathrm{bc}} =\;$$Q\left(\sqrt{\frac{2E_{\textrm{b}} 
% }{N_0 }}\right)=Q\left(\sqrt{\frac{2E_{\mathrm{tot}} }{{\mathrm{nN}}_0 }}\right)=Q\left(\sqrt{\frac{2E_{\textrm{bu}} 
% }{N_0 }\frac{k}{n}}\right)$
% 
% Assumes receiver attempts to correct upto $t$ errors
% 
% s.t. $\left(n,k\right)$ code can correct upto $t$ errors, $n$: code digits   
% $k$: information digits
% 
% $P\left(j,n\right)$= Probability of having j bit errors in n-bit sequence
% 
% Average number of errors that can happen in an n-bit sequence: ${\bar{n} 
% }_e =\sum_{j=t+1}^n j\;P\left(j,n\right)$
% 
% BER of coded systems:    $\mathrm{BER}=\frac{{\bar{n} }_{e\;} }{n}=$  $P_{\textrm{ec}}$
% 
% $$P\left(j,n\right)=\left(\overset{n}{j} \right)P_{\mathrm{bc}}^j {\left(1-P_{\textrm{bc}} 
% \right)}^{n-j}$$
% 
% therefore   ${\bar{n} }_e =\sum_{j=t+1}^n j\;\left(\overset{n}{j} \right)P_{\mathrm{bc}}^j 
% {\left(1-P_{\textrm{bc}} \right)}^{n-j}$
% 
% $$P_{\textrm{ec}} =\frac{{\bar{n} }_e }{n}=\sum_{j=t+1}^n \frac{j}{n}\left(\overset{n}{j} 
% \right)P_{\mathrm{bc}}^j {\left(1-P_{\textrm{bc}} \right)}^{n-j} \;$$
% 
% So   ${\mathrm{BER}\;\;=\;\;P}_{\textrm{ec}} \;=\sum_{j=t+1}^n \left(\overset{n-1}{j-1} 
% \right)P_{\mathrm{bc}}^j {\left(1-P_{\textrm{bc}} \right)}^{n-j}$
% 
% Theoretical values in the same graph with the simulation results
%%
Eb_N0 = -4:2:12;    %given values

%avg BER values of simulation
BER_un = [0.1637    0.1212    0.0895    0.0404    0.0143    0.0022    0.0002    0.0000         0];          %uncoded
BER_hm7_4 = [0.2342    0.1104    0.0486    0.0118    0.0012    0.0001         0         0         0];       %hamming (7,4)
BER_hm15_11 = [0.2368    0.1333    0.0709    0.0176    0.0033    0.0001    0.0000         0         0];     %hamming (15,11)

%Theoretical BER values
BER_un_th = [];
Pbc_hm7_4 = [];
Pbc_hm15_11 = [];
for i = 1:length(Eb_N0)
    EbN0 = 10^(Eb_N0(i)/10);
    BER_un_th(i) = qfunc(sqrt(2*EbN0));
    Pbc_hm7_4(i) = qfunc(sqrt(2*EbN0*(4/7)));
    Pbc_hm15_11(i) = qfunc(sqrt(2*EbN0*(11/15)));
end
BER_un_th
BER_hm7_4_th = [];
BER_hm15_11_th = [];
t=1; %hamming code is for single bit errors
%hamming (7,4)
n=7;
for i = 1:length(Eb_N0)
    a = 0;
    for j = t+1:n
        a = a + nchoosek(n-1,j-1)*(Pbc_hm7_4(i)^j)*((1-Pbc_hm7_4(i))^(n-j));
    end
    BER_hm7_4_th(i) = a;
end
BER_hm7_4_th
%hamming (15,11)
n=15;
for i = 1:length(Eb_N0)
    b = 0;
    for j = t+1:n
        b = b + nchoosek(n-1,j-1)*(Pbc_hm15_11(i)^j)*((1-Pbc_hm15_11(i))^(n-j));
    end
    BER_hm15_11_th(i) = b;
end
BER_hm15_11_th
%%
f = figure;  
f.Position = [10 10 900 800]; 

semilogy(Eb_N0,BER_un); %simulation results
hold on
semilogy(Eb_N0,BER_un_th);  %theoretical values
hold on

semilogy(Eb_N0,BER_hm7_4);
hold on
semilogy(Eb_N0,BER_hm7_4_th);
hold on

semilogy(Eb_N0,BER_hm15_11);
hold on
semilogy(Eb_N0,BER_hm15_11_th);

grid
lg = legend('Uncoded-sm','Uncoded-th','Hamming(7,4)-sm','Hamming(7,4)-th','Hamming(15,11)-sm','Hamming(15,11)-th');
lg.Location = 'southwest';
xlabel('Eb/N0 (dB)')
ylabel('Bit Error Rate')
hold off
%% 
% *Appendix*
% 
% *FUNCTIONS*
%%
%%%%%%%%%%%%%%----MATLAB R2018a-----%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--A
%%%%--A(1),(2)
function [code] = makecode(p)
[p,i] = sort(p);
[~,j] = sort(i);

if length(p)==2
    code = ["1", "0"];
    code = code(j);
else
    p(2) = p(1) + p(2); 
    p(1) = []; 
    [code] = makecode(p);
    code_prv = code(1);
    code(1) = [];
    code = [code_prv+"0",code];
    code = [code_prv+"1",code];
    code = code(j);
end
end

%%%%--A(4)
function [code_enc] = huffman_encoder(text,alphabet,hf_codes)
code_enc = '';
for i = 1 : length(text)
    ind = find(alphabet==text(i));
    code_enc = code_enc + hf_codes(ind);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--B
%%%%--B(1)
%generate CDF of probability distribution
function CDF = cdf(p)
CDF = zeros(1,length(p)+1);
CDF(1) = 0;
for i=2:length(p)+1
    CDF(i) = p(i-1) + CDF(i-1);
end
end

%generate proper sequence (X)
function X = x(s)
X = zeros(1,length(s));
for i=1:length(s)
    X(i) = str2num(s(i))+1;
end
end
%%%%--B(2)
%get tag value
function tag = basic_arithmetic_code(CDF, X)
l = 0;  %l0
u = 1;  %u0
for i = 1:length(X)
    l_prv = l;
    u_prv = u;
    l = l_prv +(u_prv - l_prv)*CDF(X(i)+1-1);
    u = l_prv +(u_prv - l_prv)*CDF(X(i)+1);
end
tag = (l+u)/2;
end
%%%%--B(3)
%Convert to binary value
function binary_codeword_tag = convert_to_binary(tag,p,X) 
l_prv=0;
for i = 1:length(X)
    l_prv = l_prv-log(p(X(i)));
end
l = ceil(l_prv) + 1;
binary_tag_array = string(fix(rem(tag*pow2(1:l),2)));
binary_codeword_tag = '';
for i=1:length(binary_tag_array)
    binary_codeword_tag = strcat(binary_codeword_tag,binary_tag_array(i));
end
end

%%%%--B(4)
%incremental encoding scheme with interval scaling
function binary_tag_incr = incremental_arithmetic_code(CDF, X)
l = 0;
u = 1;
st = '';
for i = 1:length(X)
    l_prv = l;
    u_prv = u;
    l = l_prv +(u_prv - l_prv)*CDF(X(i)+1-1);
    u = l_prv +(u_prv - l_prv)*CDF(X(i)+1);
    
    while true
        if l>=0.5
            l = 2*(l-0.5);
            u = 2*(u-0.5);
            st = strcat(st,'1');
        elseif u<0.5
            l = 2*l;
            u = 2*u;
            st = strcat(st,'0');
        else
            break
        end  
    end
end
binary_tag_incr = strcat(st,'1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--C
%%%%--C(2)
function [enc_code, stat_dic] = lzw_encoder(stat_dic, msg)
stat_dic_last = length(stat_dic);
enc_code = [];
i=1;
len=1;
while i + len <= length(msg)
    sub_msg = msg(i:i+len);
    if isKey(stat_dic,sub_msg)
            len = len + 1;
    else
        stat_dic_last = stat_dic_last + 1;
        stat_dic(sub_msg) = stat_dic_last;
        enc_code = [enc_code, stat_dic(msg(i:i+len-1))];
        i = i + len;
    end
end
len = length(msg) - i;
sub_msg = msg(i:i+len);
enc_code = [enc_code, stat_dic(sub_msg)];
end

%%%%--C(4)
function [msg_out, dic] = lzw_decoder(dic, enc_out)
dict_index_last = length(dic);
dic_begin = dic(enc_out(1));
msg_out = dic(enc_out(1));
for i=2:length(enc_out)
    if enc_out(i)==dict_index_last + 1
        dic_begin = strcat(dic_begin,dic_begin(1));
        dict_index_last = dict_index_last + 1;
        dic(dict_index_last) = dic_begin;
        temp = dic(enc_out(i));
    else
        temp = dic(enc_out(i));
        dic_begin = strcat(dic_begin,temp(1));
        dict_index_last = dict_index_last + 1;
        dic(dict_index_last) = dic_begin;
        dic_begin = dic(enc_out(i));
    end
    msg_out = strcat(msg_out,temp);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--D 
function err = error(txData, rxData)
err = 0;
for i=1:length(txData)
    if txData(i)~=rxData(i)
        err = err +1;
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--E
function deci = array_to_deci(s)
str_s = num2str(s);             %convert to a string array
str_s(isspace(str_s)) = '';     %remove the spaces
deci = bin2dec(str_s);          %convert to decimal
end

function error_out = process(k, G, H, t, snr)

txData = randi([0 1],1,k);         % Generate data
m = txData;
hc = mod(m*G,2);                   % hamming coded
modSig = pskmod(hc,2,pi/2);        % Modulate                          
rxSig = awgn(modSig,snr);              % Pass through AWGN
rxData = pskdemod(rxSig,2,pi/2);       % Demodulate
r = rxData;
syn = mod(r*H.',2);
syn_indx = array_to_deci(syn) + 1;

e = t(syn_indx,:);
corrected = xor(r, e);
error_out = error(hc, corrected);
end