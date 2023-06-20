%手动导入PCADataT.xlsx，删去第一列艺术家id和最后一列流派标签数据，命名为OD
%手动导入PCADataT.xlsx中genre_id列
%         addpath('D:\Matlab2020b\tensor_toolbox') savepath doc sptensor
%         引入张量库
%初始化先验参数：
BDI=zeros(40,2); %BD指数评估记录表格
for loopclust=4:30 %循环类数目
SampleN=90856; %样本量
ClusterN=loopclust; %类数目
IterationTimes=100; %迭代次数
%需要恢复数据时可用：
% PCADataT0=table2array(OD);
% PCADataT=PCADataT0(2:91732,:);

%Kmeans：
[IDX,C,sumd,D]=kmeans(PCADataT,ClusterN);
Alpha=rand(1,ClusterN)*0.98+0.01;
%GMM:
%初始化先验参数：
Alpha=Alpha/sum(Alpha);%混合系数向量
Mu=zeros(8,ClusterN);%均值向量组
% for i=1:20
%     Mu(:,i)=F(:,i);%+0.000001.*rand(8,1);
% end%均值向量组
Mu=C';
ClusterSampleCount=zeros(1,ClusterN);
for sample=1:SampleN
    ClusterSampleCount(1,IDX(sample,1))=ClusterSampleCount(1,IDX(sample,1))+1;
end%kmeans每类点数目
    
Sigma=zeros(8,8,ClusterN);
for cluster=1:ClusterN
    for sample=1:SampleN
        if IDX(sample,1)==cluster
           Sigma(:,:,cluster)=Sigma(:,:,cluster)+(1/ClusterSampleCount(1,cluster))*((PCADataT(sample,:)-Mu(:,cluster)')'*(PCADataT(sample,:)-Mu(:,cluster)')); 
        end
    end 
end%协方差矩阵组
Gamma=zeros(SampleN,ClusterN);%后验概率矩阵
Coe1=zeros(ClusterN,1);%PDF系数向量
InvSigma=zeros(8,8,ClusterN);%逆矩阵向量
CentralBox=zeros(SampleN,8,ClusterN);
CentralBox=CentralBox+PCADataT;
PCABaseBox=permute(CentralBox,[2 3 1]);%录入数据后的基础张量
for iteration=1:IterationTimes
    %E：
    for cluster=1:ClusterN
        Coe1(cluster,1)=(det(Sigma(:,:,cluster)))^(-0.5);%pdf
    end
    for cluster=1:ClusterN
        InvSigma(:,:,cluster)=inv(Sigma(:,:,cluster));
    end
    CentralGroup=PCABaseBox-Mu;
    CentralGroup=permute(CentralGroup,[3 1 2]);%中心矩张量
    
    ExpMat=zeros(SampleN,ClusterN);%指数矩阵
    for sample=1:SampleN
        for cluster=1:ClusterN
            ExpMat(sample,cluster)=CentralGroup(sample,:,cluster)*InvSigma(:,:,cluster)*CentralGroup(sample,:,cluster)';
        end
    end
    PdfMat=(((2*pi)^(-4))*Coe1').*(exp((-0.5)*ExpMat));%高斯pdf矩阵
    Gamma=(Alpha.*PdfMat)./(PdfMat*Alpha');%Gamma矩阵
    %M
    Mu=(PCADataT'*Gamma)./sum(Gamma);%更新均值矩阵
    CentralGroup=PCABaseBox-Mu;
    CentralGroup=permute(CentralGroup,[3 1 2]);%更新CG
    for cluster=1:ClusterN
        AvgCrossBox=zeros(8,SampleN,8);
        AvgA=AvgCrossBox+CentralGroup(:,:,cluster)';
        AvgB=permute(AvgA,[3 2 1]);
        AvgCrossGroup=AvgA.*AvgB;
        AvgCrossGroup=permute(AvgCrossGroup,[3 1 2]);
        ACG=tensor(AvgCrossGroup);
        Sigma(:,:,cluster)=double(ttv(ACG,Gamma(:,cluster),3)./sum(Gamma(:,cluster)));
    end%更新协方差矩阵
    Alpha=sum(Gamma)/SampleN;%更新混合系数向量
end
%样本归类：
ClusterVec=zeros(SampleN,1);
for sample=1:SampleN
    [tmp,ClusterVec(sample,1)]=max(Gamma(sample,:));
end
%标签对比统计联表：
CrossCount=zeros(20,ClusterN);
OriginID=GenreID.genre_id(1:SampleN);
for select=1:20
    for sample=1:SampleN
        if OriginID(sample,1)==select
            CrossCount(select,ClusterVec(sample,1))=CrossCount(select,ClusterVec(sample,1))+1;
        end
    end
end
%模型评价：
BDMy=evalclusters(PCADataT,ClusterVec,'DaviesBouldin');
BDOrigin=evalclusters(PCADataT,IDX,'DaviesBouldin');
BDI(loopclust-2,1)=BDMy.CriterionValues;
BDI(loopclust-2,2)=BDOrigin.CriterionValues;
 end
