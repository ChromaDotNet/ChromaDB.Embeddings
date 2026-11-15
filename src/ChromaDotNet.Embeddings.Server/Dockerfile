# The image of ChromaDotNet.Embeddings.Server for Linux x64 and arm64, built from the root of the repository:
#   docker buildx build --platform linux/amd64,linux/arm64 -f src/ChromaDotNet.Embeddings.Server/Dockerfile -t embeddings-server .
# The models are not in it: the service downloads them into /data at their first request; mount a volume there to keep them.
# In the container the service listens on every address, so Embeddings__ApiKey is required, or Embeddings__AllowNoApiKey=true.
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG TARGETARCH
WORKDIR /src
COPY . .
RUN dotnet publish src/ChromaDotNet.Embeddings.Server -c Release -a $TARGETARCH --self-contained false -o /out \
 && mkdir /data

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /out .
# An empty /data of the user of the image, without a RUN step, so no emulation is needed for the other architecture.
COPY --from=build --chown=$APP_UID /data /data
ENV ASPNETCORE_URLS=http://+:8080 \
    Embeddings__CacheDirectory=/data
EXPOSE 8080
VOLUME /data
USER $APP_UID
ENTRYPOINT ["dotnet", "ChromaDotNet.Embeddings.Server.dll"]
