import React from 'react'
import { ImageBackground, Text, View } from 'react-native'
import { StatusBar } from 'expo-status-bar'
const PatternBg = {uri:'https://e1.pxfuel.com/desktop-wallpaper/759/194/desktop-wallpaper-subtle-pride-phone-background-made-by-me-r-lgbt-thumbnail.jpg'}


export default function LandingPage() {
  return (
    <ImageBackground source={PatternBg} resizeMode='cover' className='h-full w-full  flex-none justify-center items-center rotate-180'>      
        <View className="h-full w-full flex-none justify-center items-center rotate-180">
            <Text className="text-white text-4xl tracking-widest" >SNAPMARK</Text>
            <StatusBar style="auto" />
        </View>
      </ImageBackground>
  )
}