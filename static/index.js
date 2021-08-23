
//function to verify wich the options were choosen, camera or video and if it's video, pass the name of the video 
function verificar(){
  var tipo=document.getElementsByName('tipo')
  var img=document.createElement("img")
  var res=document.getElementById('res')
  var tipo_drowsi=document.getElementsByName("drowsi")
  res.innerHTML=""
  var drowsi=0
  img.setAttribute("id","foto")
  if(tipo_drowsi[1].checked){
    drowsi=1
  } else if(tipo_drowsi[2].checked){
    drowsi=2
  }
  
  if(tipo[0].checked){
  
   var foto=document.getElementById("imagem")
   foto.src="camera/"+drowsi+""

  }else {
    var video=document.getElementById("videos")
    var foto=document.getElementById("imagem")
    nome_video=video.value

    foto.src="video/"+nome_video+"/"+drowsi

  }
  var button=document.getElementById("button")
  button.value="Pause"
  button.onclick=((fun)=> {return()=> fun()})(pause.bind())
 
}

//pausar video or camera
function pause(){
  var foto=document.getElementById("imagem")
  foto.src="pausar"
  var button=document.getElementById("button")
  button.value="Clique"
  button.onclick=((fun)=> {return()=> fun()})(verificar.bind())
}
